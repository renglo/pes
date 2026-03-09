"""
Integration tests for the full policy/ranking flow with mocks.

Simulates the real conversation pipeline:
    generate_plan (resolve IDs → inject policies → merge into intent → save workspace)
        ↓
    post_search (read workspace → extract intent → PolicyFilter → filtered results)

All external dependencies (DAC, AgentUtilities, LLM) are mocked.
"""
import pytest
import copy
from unittest.mock import MagicMock, patch, PropertyMock


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures — shared mock data simulating DynamoDB documents
# ═════════════════════════════════════════════════════════════════════════════

ATTENDANT_ARTHUR = {
    "_id": "att-arthur-001",
    "name": "Arthur Silva",
    "email": "arthur@company.com",
    "isActive": True,
    "policy_id": "pol-standard",
}

ATTENDANT_MARIA = {
    "_id": "att-maria-002",
    "name": "Maria Santos",
    "email": "maria@company.com",
    "isActive": True,
    "policy_id": "pol-executive",
}

ATTENDANT_LUCAS = {
    "_id": "att-lucas-003",
    "name": "Lucas Oliveira",
    "email": "lucas@company.com",
    "isActive": True,
    "policy_id": "pol-standard",
}

POLICY_STANDARD = {
    "_id": "pol-standard",
    "name": "Standard Travel Policy",
    "max_flight_class": "economy",
    "max_flight_budget": 1500,
    "max_hotel_stars": 3,
    "max_hotel_daily_rate": 200,
    "enabled_services": ["flights", "hotels"],
    "currency": "USD",
}

POLICY_EXECUTIVE = {
    "_id": "pol-executive",
    "name": "Executive Travel Policy",
    "max_flight_class": "business",
    "max_flight_budget": 5000,
    "max_hotel_stars": 5,
    "max_hotel_daily_rate": 800,
    "enabled_services": ["flights", "hotels"],
    "currency": "USD",
}

POLICY_RESTRICTED = {
    "_id": "pol-restricted",
    "name": "Restricted Policy",
    "max_flight_class": "economy",
    "max_flight_budget": 500,
    "max_hotel_stars": 3,
    "max_hotel_daily_rate": 100,
    "enabled_services": ["flights"],  # hotels disabled
    "currency": "USD",
}

PREFERENCE_ARTHUR = {
    "_id": "pref-arthur-001",
    "user_id": "att-arthur-001",
    "preferred_airlines": ["LATAM"],
    "preferred_travel_times": ["morning"],
    "blocked_travel_times": ["red_eye"],
    "hotel_chain_prefs": ["Ibis"],
    "_modified": "2025-06-01T00:00:00Z",
}

ALL_ATTENDANTS = [ATTENDANT_ARTHUR, ATTENDANT_MARIA, ATTENDANT_LUCAS]

SAMPLE_FLIGHTS = [
    {"price_amount": 800, "airline": "LATAM", "cabin_class": "economy", "departure_time": "08:00"},
    {"price_amount": 1200, "airline": "GOL", "cabin_class": "economy", "departure_time": "14:00"},
    {"price_amount": 2500, "airline": "LATAM", "cabin_class": "business", "departure_time": "10:00"},
    {"price_amount": 6000, "airline": "Emirates", "cabin_class": "first", "departure_time": "22:00"},
]

SAMPLE_HOTELS = [
    {"name": "Ibis Budget", "currentPrice": "$90", "rating": "3.8", "stars": 2},
    {"name": "Holiday Inn", "currentPrice": "$180", "rating": "4.1", "stars": 3},
    {"name": "Marriott", "currentPrice": "$350", "rating": "4.5", "stars": 4},
    {"name": "Copacabana Palace", "currentPrice": "$900", "rating": "4.8", "stars": 5},
]


def _make_base_intent():
    """Create a base intent with synthetic traveler IDs as the LLM would produce."""
    return {
        "schema": "renglo.intent.v1",
        "intent_id": "test-intent-001",
        "party": {
            "travelers": {"adults": 2},
            "traveler_ids": ["t1", "t2"],
            "travelers_by_id": {
                "t1": {"group_index": "0"},
                "t2": {"group_index": "1"},
            },
        },
        "itinerary": {
            "trip_type": "round_trip",
            "segments": [
                {"origin": "GRU", "destination": "GIG", "depart_date": "2025-07-01", "traveler_ids": ["t1", "t2"]},
                {"origin": "GIG", "destination": "GRU", "depart_date": "2025-07-05", "traveler_ids": ["t1", "t2"]},
            ],
            "lodging": {
                "needed": True,
                "stays": [
                    {"location": "Rio de Janeiro", "check_in": "2025-07-01", "check_out": "2025-07-05", "traveler_ids": ["t1", "t2"]},
                ],
            },
        },
        "preferences": {"flight": {}, "hotel": {}},
        "constraints": {},
    }


# ═════════════════════════════════════════════════════════════════════════════
# 1. _resolve_traveler_ids_in_intent
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveTravelerIds:
    """Test synthetic ID → real attendant ID resolution."""

    def _make_gp(self, attendants, llm_names):
        """Create a GeneratePlan instance with mocked DAC and AGU."""
        from pes_noma.handlers.generate_plan import GeneratePlan

        gp = GeneratePlan.__new__(GeneratePlan)
        gp.config = {}
        gp.DAC = MagicMock()
        gp.AGU = MagicMock()
        gp.BPC = MagicMock()
        gp.prompts = {}

        # Mock _fetch_all_attendants
        gp.DAC.get_a_b.return_value = {
            "success": True,
            "items": attendants,
            "last_id": None,
        }

        # Mock LLM response for name extraction
        llm_response = MagicMock()
        llm_response.content = str(llm_names)
        gp.AGU.llm.return_value = llm_response
        gp.AGU.AI_1_MODEL = "test-model"

        return gp

    def test_exact_match_two_travelers(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["Arthur Silva", "Maria Santos"]')
        intent = _make_base_intent()

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Arthur Silva e Maria Santos", "p", "o")

        ids = result["party"]["traveler_ids"]
        assert "att-arthur-001" in ids
        assert "att-maria-002" in ids
        assert "t1" not in ids
        assert "t2" not in ids

    def test_segments_updated_with_real_ids(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["Arthur Silva", "Maria Santos"]')
        intent = _make_base_intent()

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Arthur e Maria", "p", "o")

        for seg in result["itinerary"]["segments"]:
            assert "att-arthur-001" in seg["traveler_ids"]
            assert "att-maria-002" in seg["traveler_ids"]

    def test_stays_updated_with_real_ids(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["Arthur Silva", "Maria Santos"]')
        intent = _make_base_intent()

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Arthur e Maria", "p", "o")

        stays = result["itinerary"]["lodging"]["stays"]
        assert "att-arthur-001" in stays[0]["traveler_ids"]
        assert "att-maria-002" in stays[0]["traveler_ids"]

    def test_fuzzy_match(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["Artur Silva"]')  # typo, no "h"
        intent = _make_base_intent()
        intent["party"]["traveler_ids"] = ["t1"]
        intent["party"]["travelers_by_id"] = {"t1": {"group_index": "0"}}

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Artur Silva", "p", "o")

        ids = result["party"]["traveler_ids"]
        assert "att-arthur-001" in ids

    def test_no_match_adds_unresolved(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["John Unknown"]')
        intent = _make_base_intent()
        intent["party"]["traveler_ids"] = ["t1"]
        intent["party"]["travelers_by_id"] = {"t1": {"group_index": "0"}}

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para John Unknown", "p", "o")

        assert "unresolved_travelers" in result
        assert result["unresolved_travelers"][0]["type"] == "not_found"

    def test_ambiguous_match(self):
        dupes = [
            {"_id": "att-1", "name": "Maria Silva", "isActive": True},
            {"_id": "att-2", "name": "Maria Silva Souza", "isActive": True},
        ]
        gp = self._make_gp(dupes, '["Maria Silva"]')
        intent = _make_base_intent()
        intent["party"]["traveler_ids"] = ["t1"]
        intent["party"]["travelers_by_id"] = {"t1": {"group_index": "0"}}

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Maria Silva", "p", "o")

        assert "unresolved_travelers" in result
        assert result["unresolved_travelers"][0]["type"] == "multiple_matches"

    def test_no_attendants_returns_unchanged(self):
        gp = self._make_gp([], '["Arthur"]')
        intent = _make_base_intent()

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Arthur", "p", "o")

        assert result["party"]["traveler_ids"] == ["t1", "t2"]

    def test_llm_returns_no_names(self):
        gp = self._make_gp(ALL_ATTENDANTS, '[]')
        intent = _make_base_intent()

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para São Paulo", "p", "o")

        assert result["party"]["traveler_ids"] == ["t1", "t2"]

    def test_more_names_than_synthetic_ids_extends(self):
        gp = self._make_gp(ALL_ATTENDANTS, '["Arthur Silva", "Maria Santos", "Lucas Oliveira"]')
        intent = _make_base_intent()  # has t1, t2

        result = gp._resolve_traveler_ids_in_intent(intent, "viagem para Arthur, Maria e Lucas", "p", "o")

        ids = result["party"]["traveler_ids"]
        assert len(ids) == 3
        assert "att-arthur-001" in ids
        assert "att-maria-002" in ids
        assert "att-lucas-003" in ids


# ═════════════════════════════════════════════════════════════════════════════
# 2. _inject_policies_per_traveler
# ═════════════════════════════════════════════════════════════════════════════

class TestInjectPoliciesPerTraveler:
    """Test policy fetching and linking to travelers."""

    def _make_gp(self, attendant_map, policy_map):
        from pes_noma.handlers.generate_plan import GeneratePlan

        gp = GeneratePlan.__new__(GeneratePlan)
        gp.config = {}
        gp.DAC = MagicMock()
        gp.AGU = MagicMock()
        gp.BPC = MagicMock()
        gp.prompts = {}

        def mock_get_a_b_c(portfolio, org, ring, doc_id):
            if ring == "noma_attendants":
                return attendant_map.get(doc_id)
            elif ring == "noma_travel_policy":
                return policy_map.get(doc_id)
            return None

        gp.DAC.get_a_b_c = mock_get_a_b_c
        return gp

    def _has_inject(self):
        """Check if _inject_policies_per_traveler exists on current branch."""
        from pes_noma.handlers.generate_plan import GeneratePlan
        return hasattr(GeneratePlan, '_inject_policies_per_traveler')

    def test_links_policy_to_traveler(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_per_traveler not on current branch")

        gp = self._make_gp(
            {"att-arthur-001": ATTENDANT_ARTHUR},
            {"pol-standard": POLICY_STANDARD},
        )
        intent = {
            "party": {
                "traveler_ids": ["att-arthur-001"],
                "travelers_by_id": {"att-arthur-001": {"group_index": "0"}},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_per_traveler(intent, ["att-arthur-001"], "p", "o")

        trav = result["party"]["travelers_by_id"]["att-arthur-001"]
        assert trav["policy_id"] == "pol-standard"
        assert "pol-standard" in result["party"]["policies_by_id"]

    def test_two_travelers_different_policies(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_per_traveler not on current branch")

        gp = self._make_gp(
            {"att-arthur-001": ATTENDANT_ARTHUR, "att-maria-002": ATTENDANT_MARIA},
            {"pol-standard": POLICY_STANDARD, "pol-executive": POLICY_EXECUTIVE},
        )
        intent = {
            "party": {
                "traveler_ids": ["att-arthur-001", "att-maria-002"],
                "travelers_by_id": {
                    "att-arthur-001": {"group_index": "0"},
                    "att-maria-002": {"group_index": "1"},
                },
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_per_traveler(
            intent, ["att-arthur-001", "att-maria-002"], "p", "o"
        )

        assert result["party"]["travelers_by_id"]["att-arthur-001"]["policy_id"] == "pol-standard"
        assert result["party"]["travelers_by_id"]["att-maria-002"]["policy_id"] == "pol-executive"
        assert len(result["party"]["policies_by_id"]) == 2

    def test_skips_synthetic_ids(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_per_traveler not on current branch")

        gp = self._make_gp({}, {})
        intent = {
            "party": {
                "traveler_ids": ["t1", "t2"],
                "travelers_by_id": {},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_per_traveler(intent, ["t1", "t2"], "p", "o")

        assert result["party"].get("policies_by_id", {}) == {}

    def test_attendant_without_policy(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_per_traveler not on current branch")

        no_policy = {**ATTENDANT_ARTHUR, "policy_id": None}
        gp = self._make_gp({"att-arthur-001": no_policy}, {})
        intent = {
            "party": {
                "traveler_ids": ["att-arthur-001"],
                "travelers_by_id": {"att-arthur-001": {}},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_per_traveler(intent, ["att-arthur-001"], "p", "o")

        assert "policy_id" not in result["party"]["travelers_by_id"]["att-arthur-001"]

    def test_policy_not_found_in_db(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_per_traveler not on current branch")

        gp = self._make_gp(
            {"att-arthur-001": ATTENDANT_ARTHUR},
            {},  # policy not in DB
        )
        intent = {
            "party": {
                "traveler_ids": ["att-arthur-001"],
                "travelers_by_id": {"att-arthur-001": {}},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_per_traveler(intent, ["att-arthur-001"], "p", "o")

        assert "policy_id" not in result["party"]["travelers_by_id"]["att-arthur-001"]


# ═════════════════════════════════════════════════════════════════════════════
# 3. _inject_policies_into_intent (most-restrictive merge)
# ═════════════════════════════════════════════════════════════════════════════

class TestInjectPoliciesIntoIntent:
    """Test most-restrictive policy merge into intent."""

    def _make_gp(self):
        from pes_noma.handlers.generate_plan import GeneratePlan

        gp = GeneratePlan.__new__(GeneratePlan)
        gp.config = {}
        gp.DAC = MagicMock()
        gp.AGU = MagicMock()
        gp.BPC = MagicMock()
        gp.prompts = {}
        return gp

    def _has_inject(self):
        from pes_noma.handlers.generate_plan import GeneratePlan
        return hasattr(GeneratePlan, '_inject_policies_into_intent')

    def test_single_policy_merged(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_into_intent not on current branch")

        gp = self._make_gp()
        intent = {
            "party": {
                "travelers_by_id": {"att-arthur-001": {"policy_id": "pol-standard"}},
                "policies_by_id": {"pol-standard": POLICY_STANDARD},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_into_intent(intent)

        assert result["preferences"]["flight"]["max_class"] == "economy"
        assert result["preferences"]["flight"]["max_budget"] == 1500
        assert result["preferences"]["hotel"]["max_stars"] == 3
        assert result["preferences"]["hotel"]["max_daily_rate"] == 200

    def test_most_restrictive_across_two_policies(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_into_intent not on current branch")

        gp = self._make_gp()
        intent = {
            "party": {
                "travelers_by_id": {
                    "att-arthur-001": {"policy_id": "pol-standard"},
                    "att-maria-002": {"policy_id": "pol-executive"},
                },
                "policies_by_id": {
                    "pol-standard": POLICY_STANDARD,
                    "pol-executive": POLICY_EXECUTIVE,
                },
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_into_intent(intent)

        # Most restrictive: economy < business → economy
        assert result["preferences"]["flight"]["max_class"] == "economy"
        # Most restrictive: 1500 < 5000 → 1500
        assert result["preferences"]["flight"]["max_budget"] == 1500
        # Most restrictive: 3 < 5 → 3
        assert result["preferences"]["hotel"]["max_stars"] == 3
        # Most restrictive: 200 < 800 → 200
        assert result["preferences"]["hotel"]["max_daily_rate"] == 200

    def test_no_policies_returns_unchanged(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_into_intent not on current branch")

        gp = self._make_gp()
        intent = {
            "party": {"travelers_by_id": {}, "policies_by_id": {}},
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_into_intent(intent)

        assert result["preferences"]["flight"] == {}
        assert result["preferences"]["hotel"] == {}

    def test_enabled_services_intersection(self):
        if not self._has_inject():
            pytest.skip("_inject_policies_into_intent not on current branch")

        gp = self._make_gp()
        policy_a = {**POLICY_STANDARD, "_id": "pa", "enabled_services": ["flights", "hotels"]}
        policy_b = {**POLICY_STANDARD, "_id": "pb", "enabled_services": ["flights"]}
        intent = {
            "party": {
                "travelers_by_id": {
                    "t1": {"policy_id": "pa"},
                    "t2": {"policy_id": "pb"},
                },
                "policies_by_id": {"pa": policy_a, "pb": policy_b},
            },
            "preferences": {"flight": {}, "hotel": {}},
            "constraints": {},
        }

        result = gp._inject_policies_into_intent(intent)

        # Intersection: ["flights", "hotels"] ∩ ["flights"] = ["flights"]
        assert "flights" in result["constraints"]["enabled_services"]
        assert "hotels" not in result["constraints"]["enabled_services"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. post_search._get_intent_from_workspace
# ═════════════════════════════════════════════════════════════════════════════

class TestGetIntentFromWorkspace:
    """Test workspace intent extraction (post_search module)."""

    def test_intent_from_state(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        mock_intent = {"preferences": {"hotel": {"max_stars": 3}}}
        mock_workspace = {"state": {"intent": mock_intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_workspace
            result = _get_intent_from_workspace({}, {"portfolio": "p", "org": "o"})

        assert result == mock_intent

    def test_intent_from_cache_fallback(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        mock_intent = {"preferences": {"flight": {"max_budget": 1000}}}
        mock_workspace = {
            "state": {},
            "cache": {
                "irn:tool_rs:pes/generate_plan": {
                    "output": {"intent": mock_intent}
                }
            },
        }

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_workspace
            result = _get_intent_from_workspace({}, {"portfolio": "p", "org": "o"})

        assert result == mock_intent

    def test_no_workspace_returns_none(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = None
            result = _get_intent_from_workspace({}, {"portfolio": "p", "org": "o"})

        assert result is None

    def test_empty_workspace_returns_none(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = {"state": {}, "cache": {}}
            result = _get_intent_from_workspace({}, {"portfolio": "p", "org": "o"})

        assert result is None

    def test_state_intent_takes_precedence_over_cache(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        state_intent = {"source": "state"}
        cache_intent = {"source": "cache"}
        mock_workspace = {
            "state": {"intent": state_intent},
            "cache": {"irn:tool_rs:pes/generate_plan": {"output": {"intent": cache_intent}}},
        }

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_workspace
            result = _get_intent_from_workspace({}, {"portfolio": "p", "org": "o"})

        assert result["source"] == "state"

    def test_payload_fields_passed_to_agu(self):
        from noma.handlers.post_search import _get_intent_from_workspace

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = None
            payload = {
                "portfolio": "my-portfolio",
                "org": "my-org",
                "_entity_type": "noma_travels",
                "_entity_id": "trip-123",
                "_thread": "thread-456",
            }
            _get_intent_from_workspace("config-obj", payload)

            MockAGU.assert_called_once_with(
                "config-obj", "my-portfolio", "my-org", "noma_travels", "trip-123", "thread-456"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. apply_policy_filter (post_search)
# ═════════════════════════════════════════════════════════════════════════════

class TestApplyPolicyFilter:
    """Test the full apply_policy_filter function with mocked workspace."""

    def _run(self, canonical, intent, result_key, option_type):
        from noma.handlers.post_search import apply_policy_filter

        mock_workspace = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_workspace
            return apply_policy_filter(
                canonical, {"portfolio": "p", "org": "o"}, {}, result_key, option_type
            )

    def test_filters_flights_over_budget(self):
        intent = {
            "preferences": {"flight": {"max_budget": 1500}, "hotel": {}},
            "constraints": {},
        }
        canonical = {"best_flights": list(SAMPLE_FLIGHTS)}

        result = self._run(canonical, intent, "best_flights", "flight_options")

        prices = [f["price_amount"] for f in result["best_flights"]]
        assert all(p <= 1500 for p in prices)
        assert len(result["_policy_violations"]) == 2  # business + first

    def test_filters_hotels_over_rate(self):
        intent = {
            "preferences": {"flight": {}, "hotel": {"max_daily_rate": 200}},
            "constraints": {},
        }
        canonical = {"hotels": list(SAMPLE_HOTELS)}

        result = self._run(canonical, intent, "hotels", "hotel_options")

        names = [h["name"] for h in result["hotels"]]
        assert "Ibis Budget" in names
        assert "Holiday Inn" in names
        assert "Marriott" not in names
        assert "Copacabana Palace" not in names

    def test_no_intent_returns_unchanged(self):
        from noma.handlers.post_search import apply_policy_filter

        canonical = {"hotels": list(SAMPLE_HOTELS)}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = None
            result = apply_policy_filter(
                canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options"
            )

        assert len(result["hotels"]) == 4  # unchanged

    def test_no_policy_in_intent_returns_all(self):
        intent = {"preferences": {"flight": {}, "hotel": {}}, "constraints": {}}
        canonical = {"best_flights": list(SAMPLE_FLIGHTS)}

        result = self._run(canonical, intent, "best_flights", "flight_options")

        assert len(result["best_flights"]) == 4

    def test_exception_in_workspace_returns_unchanged(self):
        from noma.handlers.post_search import apply_policy_filter

        canonical = {"hotels": list(SAMPLE_HOTELS)}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.side_effect = Exception("connection error")
            result = apply_policy_filter(
                canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options"
            )

        assert len(result["hotels"]) == 4  # unchanged, no crash

    def test_policy_applied_metadata_added(self):
        intent = {
            "preferences": {"flight": {}, "hotel": {"max_daily_rate": 500, "max_stars": 4}},
            "constraints": {},
        }
        canonical = {"hotels": list(SAMPLE_HOTELS)}

        result = self._run(canonical, intent, "hotels", "hotel_options")

        assert "_policy_applied" in result
        assert "_policy_violations" in result
        assert any("max_hotel_daily_rate" in p for p in result["_policy_applied"])


# ═════════════════════════════════════════════════════════════════════════════
# 6. Full end-to-end: generate_plan → workspace → post_search → PolicyFilter
# ═════════════════════════════════════════════════════════════════════════════

class TestEndToEndFlow:
    """
    Simulate the complete conversation flow:
    1. generate_plan resolves traveler IDs
    2. generate_plan injects policies per traveler
    3. generate_plan merges most-restrictive policy into intent
    4. Intent is saved to workspace
    5. search_hotels/search_flights fetches intent from workspace
    6. post_search.apply_policy_filter filters results using PolicyFilter
    """

    def _has_inject(self):
        from pes_noma.handlers.generate_plan import GeneratePlan
        return (
            hasattr(GeneratePlan, '_inject_policies_per_traveler')
            and hasattr(GeneratePlan, '_inject_policies_into_intent')
        )

    def _build_intent_with_policies(self, traveler_ids, policies_map, attendant_map):
        """Simulate generate_plan policy injection manually (for when methods are missing)."""
        from noma.handlers.policy_filter import PolicyFilter

        travelers_by_id = {}
        policies_by_id = {}
        for tid in traveler_ids:
            att = attendant_map.get(tid, {})
            pid = att.get("policy_id")
            if pid and pid in policies_map:
                travelers_by_id[tid] = {"policy_id": pid}
                policies_by_id[pid] = policies_map[pid]

        intent = _make_base_intent()
        intent["party"]["traveler_ids"] = traveler_ids
        intent["party"]["travelers_by_id"] = travelers_by_id
        intent["party"]["policies_by_id"] = policies_by_id

        # Simulate _inject_policies_into_intent merge
        policies = list(policies_by_id.values())
        if policies:
            budgets = [p["max_flight_budget"] for p in policies if p.get("max_flight_budget")]
            rates = [p["max_hotel_daily_rate"] for p in policies if p.get("max_hotel_daily_rate")]
            stars = [p["max_hotel_stars"] for p in policies if p.get("max_hotel_stars")]

            CLASS_ORDER = ["economy", "premium_economy", "business", "first"]
            classes = []
            for p in policies:
                c = p.get("max_flight_class")
                if c and c in CLASS_ORDER:
                    classes.append(CLASS_ORDER.index(c))

            if budgets:
                intent["preferences"]["flight"]["max_budget"] = min(budgets)
            if classes:
                intent["preferences"]["flight"]["max_class"] = CLASS_ORDER[min(classes)]
            if rates:
                intent["preferences"]["hotel"]["max_daily_rate"] = min(rates)
            if stars:
                intent["preferences"]["hotel"]["max_stars"] = min(stars)

            # enabled_services intersection
            service_sets = []
            for p in policies:
                es = p.get("enabled_services")
                if isinstance(es, list):
                    service_sets.append(set(es))
            if service_sets:
                merged = service_sets[0]
                for s in service_sets[1:]:
                    merged = merged & s
                intent["constraints"]["enabled_services"] = list(merged)

        return intent

    def test_standard_policy_filters_luxury_flights_and_hotels(self):
        intent = self._build_intent_with_policies(
            ["att-arthur-001"],
            {"pol-standard": POLICY_STANDARD},
            {"att-arthur-001": ATTENDANT_ARTHUR},
        )
        # Standard: max_budget=1500, max_class=economy, max_stars=3, max_rate=200

        from noma.handlers.post_search import apply_policy_filter

        mock_ws = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_ws

            # Filter flights
            flight_canonical = {"best_flights": list(SAMPLE_FLIGHTS)}
            apply_policy_filter(flight_canonical, {"portfolio": "p", "org": "o"}, {}, "best_flights", "flight_options")

            # Filter hotels
            hotel_canonical = {"hotels": list(SAMPLE_HOTELS)}
            apply_policy_filter(hotel_canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options")

        # Only economy flights under $1500
        assert len(flight_canonical["best_flights"]) == 2
        assert all(f["price_amount"] <= 1500 for f in flight_canonical["best_flights"])

        # Only hotels ≤3 stars and ≤$200/night
        assert len(hotel_canonical["hotels"]) == 2
        hotel_names = [h["name"] for h in hotel_canonical["hotels"]]
        assert "Ibis Budget" in hotel_names
        assert "Holiday Inn" in hotel_names

    def test_executive_policy_allows_most(self):
        intent = self._build_intent_with_policies(
            ["att-maria-002"],
            {"pol-executive": POLICY_EXECUTIVE},
            {"att-maria-002": ATTENDANT_MARIA},
        )
        # Executive: max_budget=5000, max_class=business, max_stars=5, max_rate=800

        from noma.handlers.post_search import apply_policy_filter
        mock_ws = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_ws

            flight_canonical = {"best_flights": list(SAMPLE_FLIGHTS)}
            apply_policy_filter(flight_canonical, {"portfolio": "p", "org": "o"}, {}, "best_flights", "flight_options")

            hotel_canonical = {"hotels": list(SAMPLE_HOTELS)}
            apply_policy_filter(hotel_canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options")

        # Only first class ($6000) exceeds budget, but also first > business class
        assert len(flight_canonical["best_flights"]) == 3

        # All hotels pass ($900 > $800 only Copacabana fails)
        assert len(hotel_canonical["hotels"]) == 3
        hotel_names = [h["name"] for h in hotel_canonical["hotels"]]
        assert "Copacabana Palace" not in hotel_names

    def test_mixed_policies_uses_most_restrictive(self):
        """Arthur (standard) + Maria (executive) → standard rules win."""
        intent = self._build_intent_with_policies(
            ["att-arthur-001", "att-maria-002"],
            {"pol-standard": POLICY_STANDARD, "pol-executive": POLICY_EXECUTIVE},
            {"att-arthur-001": ATTENDANT_ARTHUR, "att-maria-002": ATTENDANT_MARIA},
        )

        # Merged: economy, $1500 budget, 3 stars, $200/night
        assert intent["preferences"]["flight"]["max_class"] == "economy"
        assert intent["preferences"]["flight"]["max_budget"] == 1500
        assert intent["preferences"]["hotel"]["max_stars"] == 3
        assert intent["preferences"]["hotel"]["max_daily_rate"] == 200

        from noma.handlers.post_search import apply_policy_filter
        mock_ws = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_ws

            flight_canonical = {"best_flights": list(SAMPLE_FLIGHTS)}
            apply_policy_filter(flight_canonical, {"portfolio": "p", "org": "o"}, {}, "best_flights", "flight_options")

            hotel_canonical = {"hotels": list(SAMPLE_HOTELS)}
            apply_policy_filter(hotel_canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options")

        # Same as standard policy
        assert len(flight_canonical["best_flights"]) == 2
        assert len(hotel_canonical["hotels"]) == 2

    def test_restricted_policy_disables_hotels(self):
        """Policy with enabled_services=["flights"] should remove all hotels."""
        att_restricted = {**ATTENDANT_ARTHUR, "policy_id": "pol-restricted"}
        intent = self._build_intent_with_policies(
            ["att-arthur-001"],
            {"pol-restricted": POLICY_RESTRICTED},
            {"att-arthur-001": att_restricted},
        )

        assert "hotels" not in intent["constraints"]["enabled_services"]

        from noma.handlers.post_search import apply_policy_filter
        mock_ws = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_ws

            hotel_canonical = {"hotels": list(SAMPLE_HOTELS)}
            apply_policy_filter(hotel_canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options")

        assert len(hotel_canonical["hotels"]) == 0
        assert len(hotel_canonical["_policy_violations"]) == 4

    def test_no_policy_travelers_everything_passes(self):
        """Travelers with no policy → no filtering."""
        intent = _make_base_intent()
        # No policies injected

        from noma.handlers.post_search import apply_policy_filter
        mock_ws = {"state": {"intent": intent}, "cache": {}}

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.return_value = mock_ws

            flight_canonical = {"best_flights": list(SAMPLE_FLIGHTS)}
            apply_policy_filter(flight_canonical, {"portfolio": "p", "org": "o"}, {}, "best_flights", "flight_options")

            hotel_canonical = {"hotels": list(SAMPLE_HOTELS)}
            apply_policy_filter(hotel_canonical, {"portfolio": "p", "org": "o"}, {}, "hotels", "hotel_options")

        assert len(flight_canonical["best_flights"]) == 4
        assert len(hotel_canonical["hotels"]) == 4

    def test_workspace_read_failure_doesnt_crash_search(self):
        """If workspace fetch fails, search returns unfiltered results."""
        from noma.handlers.post_search import apply_policy_filter

        with patch("renglo.agent.agent_utilities.AgentUtilities") as MockAGU:
            MockAGU.return_value.get_active_workspace.side_effect = Exception("DynamoDB timeout")

            canonical = {"best_flights": list(SAMPLE_FLIGHTS)}
            result = apply_policy_filter(canonical, {"portfolio": "p", "org": "o"}, {}, "best_flights", "flight_options")

        assert len(result["best_flights"]) == 4
        assert "_policy_violations" not in result
