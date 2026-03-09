"""
Tests for policy_filter.py (hard constraints)

Run with:
  python -m pytest extensions/pes_noma/package/pes_noma/tests/test_policy_filter.py -v
"""
import unittest
import sys
import os

# Add noma package to path (policy_filter lives in the noma extension)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'noma', 'package'))

from noma.handlers.policy_filter import (
    _parse_price,
    _flight_price,
    _hotel_rate,
    _hotel_stars,
    _flight_class,
    _class_rank,
    _check_flight,
    _check_hotel,
    _filter_list,
    PolicyFilter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

INTENT_RESTRICTIVE = {
    "preferences": {
        "flight": {"max_budget": 500.0, "max_class": "economy"},
        "hotel": {"max_daily_rate": 300.0, "max_stars": 4},
    },
    "constraints": {
        "enabled_services": ["flights", "hotels"],
    },
}

INTENT_NO_POLICY = {
    "preferences": {},
    "constraints": {},
}

INTENT_FLIGHTS_DISABLED = {
    "preferences": {},
    "constraints": {"enabled_services": ["hotels"]},
}

INTENT_HOTELS_DISABLED = {
    "preferences": {},
    "constraints": {"enabled_services": ["flights"]},
}

FLIGHT_CHEAP_ECONOMY = {
    "price": "$250",
    "cabin_class": "economy",
    "flights": [{"airline": "GOL", "departure_airport": {"time": "2026-03-10T08:00:00"}}],
}

FLIGHT_EXPENSIVE_BUSINESS = {
    "price": "$1200",
    "cabin_class": "business",
    "flights": [{"airline": "LATAM", "departure_airport": {"time": "2026-03-10T10:00:00"}}],
}

FLIGHT_OVER_BUDGET_ECONOMY = {
    "price_amount": 600,
    "cabin_class": "economy",
    "flights": [{"airline": "Azul"}],
}

FLIGHT_FIRST_CLASS = {
    "price": "$800",
    "cabin_class": "first",
    "flights": [{"airline": "Emirates"}],
}

FLIGHT_NO_CLASS = {
    "price": "$300",
    "flights": [{"airline": "Spirit"}],
}

HOTEL_CHEAP_3STAR = {
    "currentPrice": 150,
    "stars": 3,
    "name": "Budget Inn",
}

HOTEL_EXPENSIVE_5STAR = {
    "currentPrice": 500,
    "stars": 5,
    "name": "Grand Luxury Resort",
}

HOTEL_OVER_RATE = {
    "rate_per_night": 350,
    "stars": 3,
    "name": "Overpriced Motel",
}

HOTEL_OVER_STARS = {
    "currentPrice": 200,
    "star_rating": 5,
    "name": "Five Star Bargain",
}

HOTEL_NO_STARS = {
    "currentPrice": 100,
    "name": "No Rating Inn",
}


# ── Tests: helpers ────────────────────────────────────────────────────────────

class TestParsePrice(unittest.TestCase):
    def test_none(self):
        self.assertEqual(_parse_price(None), 0.0)

    def test_int(self):
        self.assertEqual(_parse_price(300), 300.0)

    def test_float(self):
        self.assertEqual(_parse_price(299.99), 299.99)

    def test_dollar_string(self):
        self.assertAlmostEqual(_parse_price("$1,234.56"), 1234.56)

    def test_plain_string(self):
        self.assertAlmostEqual(_parse_price("450"), 450.0)

    def test_no_number(self):
        self.assertEqual(_parse_price("free"), 0.0)


class TestFlightPrice(unittest.TestCase):
    def test_price_amount(self):
        self.assertEqual(_flight_price({"price_amount": 350}), 350.0)

    def test_extracted_price(self):
        self.assertEqual(_flight_price({"extracted_price": 400}), 400.0)

    def test_string_price(self):
        self.assertAlmostEqual(_flight_price({"price": "$250"}), 250.0)

    def test_price_amount_priority(self):
        self.assertEqual(_flight_price({"price_amount": 100, "price": "$999"}), 100.0)


class TestHotelRate(unittest.TestCase):
    def test_rate_per_night(self):
        self.assertEqual(_hotel_rate({"rate_per_night": 200}), 200.0)

    def test_current_price(self):
        self.assertEqual(_hotel_rate({"currentPrice": 150}), 150.0)

    def test_dict_rate(self):
        self.assertEqual(_hotel_rate({"rate_per_night": {"lowest": 180}}), 180.0)


class TestHotelStars(unittest.TestCase):
    def test_stars(self):
        self.assertEqual(_hotel_stars({"stars": 4}), 4.0)

    def test_star_rating(self):
        self.assertEqual(_hotel_stars({"star_rating": 3.5}), 3.5)

    def test_none(self):
        self.assertIsNone(_hotel_stars({}))

    def test_invalid(self):
        self.assertIsNone(_hotel_stars({"stars": "N/A"}))


class TestFlightClass(unittest.TestCase):
    def test_cabin_class(self):
        self.assertEqual(_flight_class({"cabin_class": "Business"}), "business")

    def test_travel_class(self):
        self.assertEqual(_flight_class({"travel_class": "Premium Economy"}), "premium_economy")

    def test_none(self):
        self.assertIsNone(_flight_class({}))


class TestClassRank(unittest.TestCase):
    def test_order(self):
        self.assertLess(_class_rank("economy"), _class_rank("business"))
        self.assertLess(_class_rank("business"), _class_rank("first"))
        self.assertLess(_class_rank("premium_economy"), _class_rank("business"))

    def test_unknown(self):
        self.assertEqual(_class_rank("unknown_class"), 0)

    def test_none(self):
        self.assertEqual(_class_rank(None), 0)


# ── Tests: check functions ───────────────────────────────────────────────────

class TestCheckFlight(unittest.TestCase):
    def test_compliant(self):
        self.assertIsNone(_check_flight(FLIGHT_CHEAP_ECONOMY, 500, "economy"))

    def test_over_budget(self):
        reason = _check_flight(FLIGHT_OVER_BUDGET_ECONOMY, 500, "economy")
        self.assertIsNotNone(reason)
        self.assertIn("exceeds max_flight_budget", reason)

    def test_over_class(self):
        reason = _check_flight(FLIGHT_EXPENSIVE_BUSINESS, 2000, "economy")
        self.assertIsNotNone(reason)
        self.assertIn("exceeds max_flight_class", reason)

    def test_first_class_blocked(self):
        reason = _check_flight(FLIGHT_FIRST_CLASS, 2000, "business")
        self.assertIsNotNone(reason)
        self.assertIn("first", reason)

    def test_no_max_budget(self):
        self.assertIsNone(_check_flight(FLIGHT_EXPENSIVE_BUSINESS, None, None))

    def test_no_class_info(self):
        self.assertIsNone(_check_flight(FLIGHT_NO_CLASS, 500, "economy"))

    def test_both_violations(self):
        reason = _check_flight(FLIGHT_EXPENSIVE_BUSINESS, 500, "economy")
        self.assertIsNotNone(reason)
        self.assertIn("exceeds", reason)


class TestCheckHotel(unittest.TestCase):
    def test_compliant(self):
        self.assertIsNone(_check_hotel(HOTEL_CHEAP_3STAR, 300, 4))

    def test_over_rate(self):
        reason = _check_hotel(HOTEL_OVER_RATE, 300, 5)
        self.assertIsNotNone(reason)
        self.assertIn("exceeds max_hotel_daily_rate", reason)

    def test_over_stars(self):
        reason = _check_hotel(HOTEL_OVER_STARS, 500, 4)
        self.assertIsNotNone(reason)
        self.assertIn("exceeds max_hotel_stars", reason)

    def test_no_limits(self):
        self.assertIsNone(_check_hotel(HOTEL_EXPENSIVE_5STAR, None, None))

    def test_no_stars_info(self):
        self.assertIsNone(_check_hotel(HOTEL_NO_STARS, 500, 4))


# ── Tests: filter_list ───────────────────────────────────────────────────────

class TestFilterList(unittest.TestCase):
    def test_all_pass(self):
        opts = [FLIGHT_CHEAP_ECONOMY, FLIGHT_NO_CLASS]
        compliant, violations = _filter_list(opts, lambda o: _check_flight(o, 500, "economy"))
        self.assertEqual(len(compliant), 2)
        self.assertEqual(len(violations), 0)

    def test_all_fail(self):
        opts = [FLIGHT_EXPENSIVE_BUSINESS, FLIGHT_FIRST_CLASS]
        compliant, violations = _filter_list(opts, lambda o: _check_flight(o, 500, "economy"))
        self.assertEqual(len(compliant), 0)
        self.assertEqual(len(violations), 2)
        self.assertIn("_policy_violation", violations[0])

    def test_mixed(self):
        opts = [FLIGHT_CHEAP_ECONOMY, FLIGHT_OVER_BUDGET_ECONOMY]
        compliant, violations = _filter_list(opts, lambda o: _check_flight(o, 500, "economy"))
        self.assertEqual(len(compliant), 1)
        self.assertEqual(len(violations), 1)

    def test_empty_list(self):
        compliant, violations = _filter_list([], lambda o: None)
        self.assertEqual(len(compliant), 0)
        self.assertEqual(len(violations), 0)


# ── Tests: PolicyFilter.run() ────────────────────────────────────────────────

class TestPolicyFilterRun(unittest.TestCase):
    def setUp(self):
        self.pf = PolicyFilter()

    def test_restrictive_policy_filters_flights(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [FLIGHT_CHEAP_ECONOMY, FLIGHT_EXPENSIVE_BUSINESS, FLIGHT_OVER_BUDGET_ECONOMY],
            "hotel_options": [],
        })
        self.assertTrue(result["success"])
        out = result["output"]
        self.assertEqual(len(out["flight_options"]), 1)
        self.assertEqual(out["violations_count"], 2)

    def test_restrictive_policy_filters_hotels(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [],
            "hotel_options": [HOTEL_CHEAP_3STAR, HOTEL_EXPENSIVE_5STAR, HOTEL_OVER_RATE, HOTEL_OVER_STARS],
        })
        self.assertTrue(result["success"])
        out = result["output"]
        self.assertEqual(len(out["hotel_options"]), 1)
        self.assertEqual(out["violations_count"], 3)

    def test_no_policy_keeps_all(self):
        result = self.pf.run({
            "intent": INTENT_NO_POLICY,
            "flight_options": [FLIGHT_CHEAP_ECONOMY, FLIGHT_EXPENSIVE_BUSINESS, FLIGHT_FIRST_CLASS],
            "hotel_options": [HOTEL_CHEAP_3STAR, HOTEL_EXPENSIVE_5STAR],
        })
        out = result["output"]
        self.assertEqual(len(out["flight_options"]), 3)
        self.assertEqual(len(out["hotel_options"]), 2)
        self.assertEqual(out["violations_count"], 0)

    def test_flights_disabled(self):
        result = self.pf.run({
            "intent": INTENT_FLIGHTS_DISABLED,
            "flight_options": [FLIGHT_CHEAP_ECONOMY, FLIGHT_EXPENSIVE_BUSINESS],
            "hotel_options": [HOTEL_CHEAP_3STAR],
        })
        out = result["output"]
        self.assertEqual(len(out["flight_options"]), 0)
        self.assertEqual(len(out["hotel_options"]), 1)
        self.assertIn("flights disabled by policy", out["policy_applied"])

    def test_hotels_disabled(self):
        result = self.pf.run({
            "intent": INTENT_HOTELS_DISABLED,
            "flight_options": [FLIGHT_CHEAP_ECONOMY],
            "hotel_options": [HOTEL_CHEAP_3STAR, HOTEL_EXPENSIVE_5STAR],
        })
        out = result["output"]
        self.assertEqual(len(out["flight_options"]), 1)
        self.assertEqual(len(out["hotel_options"]), 0)
        self.assertIn("hotels disabled by policy", out["policy_applied"])

    def test_multi_segment_filtering(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [],
            "hotel_options": [],
            "flight_options_by_segment": [
                [FLIGHT_CHEAP_ECONOMY, FLIGHT_EXPENSIVE_BUSINESS],
                [FLIGHT_OVER_BUDGET_ECONOMY, FLIGHT_NO_CLASS],
            ],
            "hotel_options_by_stay": [],
        })
        out = result["output"]
        self.assertEqual(len(out["flight_options_by_segment"][0]), 1)
        self.assertEqual(len(out["flight_options_by_segment"][1]), 1)
        self.assertEqual(out["violations_count"], 2)

    def test_multi_stay_filtering(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [],
            "hotel_options": [],
            "flight_options_by_segment": [],
            "hotel_options_by_stay": [
                [HOTEL_CHEAP_3STAR, HOTEL_EXPENSIVE_5STAR],
                [HOTEL_OVER_RATE, HOTEL_NO_STARS],
            ],
        })
        out = result["output"]
        self.assertEqual(len(out["hotel_options_by_stay"][0]), 1)
        self.assertEqual(len(out["hotel_options_by_stay"][1]), 1)
        self.assertEqual(out["violations_count"], 2)

    def test_policy_applied_list(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [FLIGHT_CHEAP_ECONOMY],
            "hotel_options": [HOTEL_CHEAP_3STAR],
        })
        applied = result["output"]["policy_applied"]
        self.assertTrue(any("max_flight_budget" in r for r in applied))

    def test_empty_intent(self):
        result = self.pf.run({
            "intent": {},
            "flight_options": [FLIGHT_EXPENSIVE_BUSINESS],
            "hotel_options": [HOTEL_EXPENSIVE_5STAR],
        })
        out = result["output"]
        self.assertEqual(len(out["flight_options"]), 1)
        self.assertEqual(len(out["hotel_options"]), 1)
        self.assertEqual(out["violations_count"], 0)

    def test_check_missing_intent(self):
        result = self.pf.check({})
        self.assertFalse(result["success"])

    def test_check_valid(self):
        result = self.pf.check({"intent": INTENT_RESTRICTIVE})
        self.assertTrue(result["success"])

    def test_violations_have_reason(self):
        result = self.pf.run({
            "intent": INTENT_RESTRICTIVE,
            "flight_options": [FLIGHT_EXPENSIVE_BUSINESS],
            "hotel_options": [],
        })
        violations = result["output"]["violations"]
        self.assertEqual(len(violations), 1)
        self.assertIn("_policy_violation", violations[0])
        self.assertIn("exceeds", violations[0]["_policy_violation"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
