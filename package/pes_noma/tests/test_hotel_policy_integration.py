"""
Tests for PolicyFilter integration with hotel search results.

Validates that hotel options returned by search_hotels (Google Hotels API format)
are correctly filtered against travel policy constraints (max_daily_rate, max_stars,
enabled_services, multi-stay).
"""
import pytest
from noma.handlers.policy_filter import (
    PolicyFilter,
    _hotel_rate,
    _hotel_stars,
    _check_hotel,
    _filter_list,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_hotel(
    name="Hotel Test",
    current_price="$150",
    rating="4.2",
    stars=None,
    rate_per_night=None,
    amenities=None,
    area="Downtown",
    **extra,
):
    """Build a hotel dict in Google Hotels canonical format."""
    h = {
        "id": f"token-{name.lower().replace(' ', '-')}",
        "name": name,
        "image": "https://img.example.com/photo.jpg",
        "thumbnail": "https://img.example.com/thumb.jpg",
        "rating": rating,
        "amenities": amenities or ["Free Wi-Fi", "Pool"],
        "currentPrice": current_price,
        "nights": 3,
        "adults": "2",
        "description": f"{name} description",
        "latitude": "-22.9068",
        "longitude": "-43.1729",
        "roomType": "Standard Room",
        "area": area,
        "check_in_date": "2025-07-01",
        "check_out_date": "2025-07-04",
        "reviewCount": 120,
    }
    if stars is not None:
        h["stars"] = stars
    if rate_per_night is not None:
        h["rate_per_night"] = rate_per_night
    h.update(extra)
    return h


def make_intent(max_daily_rate=None, max_stars=None, enabled_services=None, max_budget=None, max_class=None):
    """Build a minimal intent dict with hotel policy rules."""
    intent = {"preferences": {"flight": {}, "hotel": {}}, "constraints": {}}
    if max_daily_rate is not None:
        intent["preferences"]["hotel"]["max_daily_rate"] = max_daily_rate
    if max_stars is not None:
        intent["preferences"]["hotel"]["max_stars"] = max_stars
    if max_budget is not None:
        intent["preferences"]["flight"]["max_budget"] = max_budget
    if max_class is not None:
        intent["preferences"]["flight"]["max_class"] = max_class
    if enabled_services is not None:
        intent["constraints"]["enabled_services"] = enabled_services
    return intent


# ═════════════════════════════════════════════════════════════════════════════
# 1. _hotel_rate — price extraction from various hotel formats
# ═════════════════════════════════════════════════════════════════════════════

class TestHotelRateExtraction:
    """Test _hotel_rate with all Google Hotels price formats."""

    def test_current_price_dollar_string(self):
        h = make_hotel(current_price="$150")
        assert _hotel_rate(h) == 150.0

    def test_current_price_brl_string(self):
        h = make_hotel(current_price="R$ 450")
        assert _hotel_rate(h) == 450.0

    def test_current_price_euro_string(self):
        h = make_hotel(current_price="€220")
        assert _hotel_rate(h) == 220.0

    def test_current_price_numeric(self):
        h = make_hotel(current_price=180)
        assert _hotel_rate(h) == 180.0

    def test_current_price_float(self):
        h = make_hotel(current_price=199.99)
        assert _hotel_rate(h) == 199.99

    def test_current_price_with_comma_thousands(self):
        h = make_hotel(current_price="$1,250")
        assert _hotel_rate(h) == 1250.0

    def test_rate_per_night_takes_precedence(self):
        h = make_hotel(current_price="$999", rate_per_night=120)
        assert _hotel_rate(h) == 120.0

    def test_rate_per_night_dict_lowest(self):
        h = {"rate_per_night": {"lowest": "$95", "extracted": 100}}
        assert _hotel_rate(h) == 95.0

    def test_rate_per_night_dict_extracted(self):
        h = {"rate_per_night": {"extracted": 110}}
        assert _hotel_rate(h) == 110.0

    def test_price_field_fallback(self):
        h = {"price": "200"}
        assert _hotel_rate(h) == 200.0

    def test_no_price_returns_zero(self):
        h = {"name": "No Price Hotel"}
        assert _hotel_rate(h) == 0.0

    def test_empty_string_price(self):
        h = make_hotel(current_price="")
        assert _hotel_rate(h) == 0.0

    def test_none_price(self):
        h = make_hotel(current_price=None)
        # rate_per_night is also None, price is also None → 0
        h.pop("currentPrice", None)
        assert _hotel_rate(h) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 2. _hotel_stars — star rating extraction
# ═════════════════════════════════════════════════════════════════════════════

class TestHotelStarsExtraction:

    def test_stars_integer(self):
        h = {"stars": 5}
        assert _hotel_stars(h) == 5.0

    def test_stars_float(self):
        h = {"stars": 4.5}
        assert _hotel_stars(h) == 4.5

    def test_stars_string(self):
        h = {"stars": "3"}
        assert _hotel_stars(h) == 3.0

    def test_star_rating_field(self):
        h = {"star_rating": 4}
        assert _hotel_stars(h) == 4.0

    def test_rating_field_fallback(self):
        """Google Hotels uses 'rating' for location rating; it's used as fallback."""
        h = {"rating": "4.2"}
        assert _hotel_stars(h) == 4.2

    def test_no_stars_returns_none(self):
        h = {"name": "Unrated Hotel"}
        assert _hotel_stars(h) is None

    def test_invalid_stars_returns_none(self):
        h = {"stars": "N/A"}
        assert _hotel_stars(h) is None

    def test_stars_zero(self):
        """Zero is falsy but could be a valid value — currently returns None due to `or` chain."""
        h = {"stars": 0}
        # 0 is falsy → falls through to star_rating → rating → None
        assert _hotel_stars(h) is None


# ═════════════════════════════════════════════════════════════════════════════
# 3. _check_hotel — individual hotel compliance check
# ═════════════════════════════════════════════════════════════════════════════

class TestCheckHotel:

    def test_compliant_hotel_no_limits(self):
        h = make_hotel(current_price="$200", stars=4)
        assert _check_hotel(h, None, None) is None

    def test_compliant_under_rate(self):
        h = make_hotel(current_price="$150")
        assert _check_hotel(h, max_daily_rate=200, max_stars=None) is None

    def test_compliant_exact_rate(self):
        h = make_hotel(current_price="$200")
        assert _check_hotel(h, max_daily_rate=200, max_stars=None) is None

    def test_violates_rate(self):
        h = make_hotel(current_price="$350")
        reason = _check_hotel(h, max_daily_rate=300, max_stars=None)
        assert reason is not None
        assert "350" in reason
        assert "300" in reason

    def test_compliant_under_stars(self):
        h = make_hotel(stars=3)
        assert _check_hotel(h, max_daily_rate=None, max_stars=4) is None

    def test_compliant_exact_stars(self):
        h = make_hotel(stars=4)
        assert _check_hotel(h, max_daily_rate=None, max_stars=4) is None

    def test_violates_stars(self):
        h = make_hotel(stars=5)
        reason = _check_hotel(h, max_daily_rate=None, max_stars=4)
        assert reason is not None
        assert "5" in reason
        assert "4" in reason

    def test_violates_both_rate_returns_first(self):
        """Rate is checked first, so rate violation is returned."""
        h = make_hotel(current_price="$500", stars=5)
        reason = _check_hotel(h, max_daily_rate=300, max_stars=4)
        assert "rate" in reason.lower() or "500" in reason

    def test_no_stars_field_always_compliant_on_stars(self):
        """Hotel without stars/star_rating/rating fields has no star check."""
        h = make_hotel(current_price="$100", rating=None)
        # Remove rating so _hotel_stars returns None
        h.pop("rating", None)
        assert _check_hotel(h, max_daily_rate=None, max_stars=3) is None

    def test_zero_price_always_compliant(self):
        """Hotels with price=0 are not filtered (price > 0 check)."""
        h = make_hotel(current_price="$0")
        assert _check_hotel(h, max_daily_rate=50, max_stars=None) is None

    def test_luxury_hotel_under_budget(self):
        h = make_hotel(current_price="$800", stars=5)
        assert _check_hotel(h, max_daily_rate=1000, max_stars=5) is None

    def test_budget_hotel_no_constraints(self):
        h = make_hotel(current_price="$30", stars=2)
        assert _check_hotel(h, max_daily_rate=None, max_stars=None) is None


# ═════════════════════════════════════════════════════════════════════════════
# 4. PolicyFilter.run() — full hotel filtering via intent
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyFilterHotels:

    def test_no_policy_returns_all_hotels(self):
        hotels = [make_hotel(name=f"H{i}", current_price=f"${100+i*50}") for i in range(5)]
        result = PolicyFilter().run({"intent": make_intent(), "hotel_options": hotels})
        assert result["success"]
        assert len(result["output"]["hotel_options"]) == 5
        assert result["output"]["violations"] == []

    def test_max_daily_rate_filters_expensive(self):
        hotels = [
            make_hotel(name="Budget", current_price="$80"),
            make_hotel(name="Mid", current_price="$200"),
            make_hotel(name="Luxury", current_price="$500"),
            make_hotel(name="Ultra", current_price="$1200"),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=250),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 2
        names = [h["name"] for h in out["hotel_options"]]
        assert "Budget" in names
        assert "Mid" in names
        assert len(out["violations"]) == 2

    def test_max_stars_filters_luxury(self):
        hotels = [
            make_hotel(name="2-Star", stars=2),
            make_hotel(name="3-Star", stars=3),
            make_hotel(name="4-Star", stars=4),
            make_hotel(name="5-Star", stars=5),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_stars=3),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 2
        names = [h["name"] for h in out["hotel_options"]]
        assert "2-Star" in names
        assert "3-Star" in names
        assert len(out["violations"]) == 2

    def test_combined_rate_and_stars(self):
        hotels = [
            make_hotel(name="Cheap3Star", current_price="$100", stars=3),
            make_hotel(name="Cheap5Star", current_price="$100", stars=5),  # violates stars
            make_hotel(name="Expensive3Star", current_price="$600", stars=3),  # violates rate
            make_hotel(name="Expensive5Star", current_price="$600", stars=5),  # violates both
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=300, max_stars=4),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 1
        assert out["hotel_options"][0]["name"] == "Cheap3Star"
        assert len(out["violations"]) == 3

    def test_all_hotels_removed(self):
        hotels = [
            make_hotel(name="H1", current_price="$500"),
            make_hotel(name="H2", current_price="$400"),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=100),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 0
        assert len(out["violations"]) == 2

    def test_no_hotels_empty_input(self):
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=200),
            "hotel_options": [],
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 0
        assert len(out["violations"]) == 0

    def test_hotels_disabled_by_enabled_services(self):
        hotels = [
            make_hotel(name="H1", current_price="$100"),
            make_hotel(name="H2", current_price="$200"),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["flights"]),  # hotels NOT in list
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 0
        assert len(out["violations"]) == 2
        assert any("hotels not in enabled_services" in v["_policy_violation"] for v in out["violations"])
        assert "hotels disabled by policy" in out["policy_applied"]

    def test_hotels_enabled_in_services(self):
        hotels = [make_hotel(name="H1", current_price="$100")]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["flights", "hotels"]),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 1

    def test_hotels_only_service(self):
        hotels = [make_hotel(name="H1", current_price="$100")]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["hotels"]),
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["hotel_options"]) == 1

    def test_violation_contains_reason_string(self):
        hotels = [make_hotel(name="Overpriced", current_price="$999")]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=200),
            "hotel_options": hotels,
        })
        v = result["output"]["violations"][0]
        assert "_policy_violation" in v
        assert "999" in v["_policy_violation"]
        assert "200" in v["_policy_violation"]

    def test_violation_preserves_hotel_data(self):
        h = make_hotel(name="Fancy Hotel", current_price="$800", area="Copacabana")
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=300),
            "hotel_options": [h],
        })
        v = result["output"]["violations"][0]
        assert v["name"] == "Fancy Hotel"
        assert v["area"] == "Copacabana"

    def test_policy_applied_includes_rate_rule(self):
        hotels = [make_hotel(current_price="$100")]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=500),
            "hotel_options": hotels,
        })
        assert any("max_hotel_daily_rate" in p for p in result["output"]["policy_applied"])

    def test_policy_applied_includes_stars_rule(self):
        hotels = [make_hotel(stars=3)]
        result = PolicyFilter().run({
            "intent": make_intent(max_stars=4),
            "hotel_options": hotels,
        })
        assert any("max_hotel_stars" in p for p in result["output"]["policy_applied"])


# ═════════════════════════════════════════════════════════════════════════════
# 5. Multi-stay hotel filtering
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyFilterMultiStay:

    def test_multi_stay_filters_per_stay(self):
        stay1 = [
            make_hotel(name="S1-Cheap", current_price="$100"),
            make_hotel(name="S1-Expensive", current_price="$600"),
        ]
        stay2 = [
            make_hotel(name="S2-Cheap", current_price="$80"),
            make_hotel(name="S2-Mid", current_price="$250"),
            make_hotel(name="S2-Expensive", current_price="$900"),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=300),
            "hotel_options_by_stay": [stay1, stay2],
        })
        out = result["output"]
        assert len(out["hotel_options_by_stay"]) == 2
        assert len(out["hotel_options_by_stay"][0]) == 1
        assert out["hotel_options_by_stay"][0][0]["name"] == "S1-Cheap"
        assert len(out["hotel_options_by_stay"][1]) == 2
        names_stay2 = [h["name"] for h in out["hotel_options_by_stay"][1]]
        assert "S2-Cheap" in names_stay2
        assert "S2-Mid" in names_stay2
        assert len(out["violations"]) == 2

    def test_multi_stay_all_filtered_one_stay(self):
        stay1 = [make_hotel(name="OK", current_price="$100")]
        stay2 = [make_hotel(name="Nope", current_price="$999")]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=200),
            "hotel_options_by_stay": [stay1, stay2],
        })
        out = result["output"]
        assert len(out["hotel_options_by_stay"][0]) == 1
        assert len(out["hotel_options_by_stay"][1]) == 0

    def test_multi_stay_empty_stays(self):
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=200),
            "hotel_options_by_stay": [[], []],
        })
        out = result["output"]
        assert out["hotel_options_by_stay"] == [[], []]

    def test_multi_stay_disabled_by_services(self):
        stay1 = [make_hotel(name="H1", current_price="$100")]
        stay2 = [make_hotel(name="H2", current_price="$200")]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["flights"]),
            "hotel_options_by_stay": [stay1, stay2],
        })
        out = result["output"]
        assert out["hotel_options_by_stay"] == []
        assert len(out["violations"]) == 2

    def test_multi_stay_stars_filter(self):
        stay1 = [
            make_hotel(name="3Star", stars=3),
            make_hotel(name="5Star", stars=5),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_stars=4),
            "hotel_options_by_stay": [stay1],
        })
        out = result["output"]
        assert len(out["hotel_options_by_stay"][0]) == 1
        assert out["hotel_options_by_stay"][0][0]["name"] == "3Star"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Mixed flights + hotels in single PolicyFilter.run() call
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyFilterMixed:

    def test_flights_and_hotels_filtered_independently(self):
        flights = [
            {"price_amount": 300, "airline": "LATAM", "cabin_class": "economy"},
            {"price_amount": 2000, "airline": "GOL", "cabin_class": "business"},
        ]
        hotels = [
            make_hotel(name="Cheap", current_price="$100"),
            make_hotel(name="Expensive", current_price="$800"),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_budget=500, max_daily_rate=300),
            "flight_options": flights,
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["flight_options"]) == 1
        assert len(out["hotel_options"]) == 1
        assert out["flight_options"][0]["airline"] == "LATAM"
        assert out["hotel_options"][0]["name"] == "Cheap"
        assert len(out["violations"]) == 2

    def test_flights_disabled_hotels_kept(self):
        flights = [{"price_amount": 100, "cabin_class": "economy"}]
        hotels = [make_hotel(name="H1", current_price="$100")]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["hotels"]),
            "flight_options": flights,
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["flight_options"]) == 0
        assert len(out["hotel_options"]) == 1

    def test_hotels_disabled_flights_kept(self):
        flights = [{"price_amount": 100, "cabin_class": "economy"}]
        hotels = [make_hotel(name="H1", current_price="$100")]
        result = PolicyFilter().run({
            "intent": make_intent(enabled_services=["flights"]),
            "flight_options": flights,
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["flight_options"]) == 1
        assert len(out["hotel_options"]) == 0

    def test_no_policy_keeps_everything(self):
        flights = [{"price_amount": 5000, "cabin_class": "first"}]
        hotels = [make_hotel(name="Palace", current_price="$2000", stars=5)]
        result = PolicyFilter().run({
            "intent": make_intent(),
            "flight_options": flights,
            "hotel_options": hotels,
        })
        out = result["output"]
        assert len(out["flight_options"]) == 1
        assert len(out["hotel_options"]) == 1
        assert out["violations"] == []


# ═════════════════════════════════════════════════════════════════════════════
# 7. Google Hotels canonical format edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestGoogleHotelsEdgeCases:

    def test_hotel_with_all_amenities(self):
        h = make_hotel(
            name="Full Amenity Hotel",
            current_price="$200",
            stars=4,
            amenities=["Free Wi-Fi", "Pool", "Gym", "Spa", "Restaurant", "Bar", "Parking"],
        )
        assert _check_hotel(h, max_daily_rate=300, max_stars=5) is None

    def test_hotel_missing_most_fields(self):
        """Minimal hotel dict — only name and price."""
        h = {"name": "Bare Minimum", "currentPrice": "$100"}
        assert _check_hotel(h, max_daily_rate=200, max_stars=5) is None

    def test_hotel_with_special_chars_in_price(self):
        h = make_hotel(current_price="US$ 350.00")
        assert _hotel_rate(h) == 350.0

    def test_hotel_price_with_spaces(self):
        h = make_hotel(current_price="  $ 275  ")
        assert _hotel_rate(h) == 275.0

    def test_hotel_with_numeric_string_rating(self):
        h = {"stars": "4.5"}
        assert _hotel_stars(h) == 4.5

    def test_many_hotels_performance(self):
        """50 hotels should filter in well under 50ms."""
        import time
        hotels = [make_hotel(name=f"H{i}", current_price=f"${50 + i*10}", stars=2 + (i % 4)) for i in range(50)]
        intent = make_intent(max_daily_rate=300, max_stars=4)
        start = time.perf_counter()
        result = PolicyFilter().run({"intent": intent, "hotel_options": hotels})
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert result["success"]
        assert elapsed_ms < 50, f"Took {elapsed_ms:.1f}ms for 50 hotels"

    def test_realistic_google_hotels_response(self):
        """Simulate a real Google Hotels API response structure."""
        hotels = [
            {
                "id": "ChYIyJfP0oeXg4NQGA",
                "name": "Copacabana Palace",
                "image": "https://lh5.googleusercontent.com/photo.jpg",
                "thumbnail": "https://lh5.googleusercontent.com/thumb.jpg",
                "rating": "4.7",
                "amenities": ["Free Wi-Fi", "Pool", "Spa", "Restaurant", "Bar", "Gym", "Beach access"],
                "currentPrice": "$850",
                "nights": 3,
                "adults": "2",
                "description": "Iconic luxury hotel on Copacabana Beach",
                "latitude": "-22.9668",
                "longitude": "-43.1789",
                "roomType": "Superior Room",
                "area": "Copacabana",
                "check_in_date": "2025-07-01",
                "check_out_date": "2025-07-04",
                "reviewCount": 3452,
            },
            {
                "id": "ChYIz8mT2pfRstHdARAB",
                "name": "Ibis Copacabana",
                "image": "https://lh5.googleusercontent.com/photo2.jpg",
                "thumbnail": "https://lh5.googleusercontent.com/thumb2.jpg",
                "rating": "4.1",
                "amenities": ["Free Wi-Fi", "Air conditioning"],
                "currentPrice": "$120",
                "nights": 3,
                "adults": "2",
                "description": "Budget-friendly hotel near Copacabana",
                "latitude": "-22.9670",
                "longitude": "-43.1785",
                "roomType": "Standard Room",
                "area": "Copacabana",
                "check_in_date": "2025-07-01",
                "check_out_date": "2025-07-04",
                "reviewCount": 890,
            },
            {
                "id": "ChYI5v6K3ZaJx_RdEAE",
                "name": "Hotel Atlantico Business",
                "image": "https://lh5.googleusercontent.com/photo3.jpg",
                "thumbnail": "https://lh5.googleusercontent.com/thumb3.jpg",
                "rating": "4.3",
                "amenities": ["Free Wi-Fi", "Pool", "Restaurant", "Gym"],
                "currentPrice": "$250",
                "nights": 3,
                "adults": "2",
                "description": "Business hotel in Centro",
                "latitude": "-22.9083",
                "longitude": "-43.1764",
                "roomType": "Business Room",
                "area": "Centro",
                "check_in_date": "2025-07-01",
                "check_out_date": "2025-07-04",
                "reviewCount": 1200,
            },
        ]
        # Policy: max R$300/night
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=300),
            "hotel_options": hotels,
        })
        out = result["output"]
        names = [h["name"] for h in out["hotel_options"]]
        assert "Copacabana Palace" not in names  # $850 > $300
        assert "Ibis Copacabana" in names  # $120 OK
        assert "Hotel Atlantico Business" in names  # $250 OK
        assert len(out["violations"]) == 1
        assert out["violations"][0]["name"] == "Copacabana Palace"

    def test_realistic_corporate_policy_3star_max(self):
        """Corporate policy: max 3 stars, max $200/night."""
        hotels = [
            make_hotel(name="Hilton", current_price="$350", stars=5),
            make_hotel(name="Holiday Inn", current_price="$180", stars=3),
            make_hotel(name="Ibis", current_price="$90", stars=2),
            make_hotel(name="Marriott", current_price="$280", stars=4),
            make_hotel(name="Hostel Central", current_price="$40", stars=1),
        ]
        result = PolicyFilter().run({
            "intent": make_intent(max_daily_rate=200, max_stars=3),
            "hotel_options": hotels,
        })
        out = result["output"]
        names = [h["name"] for h in out["hotel_options"]]
        assert names == ["Holiday Inn", "Ibis", "Hostel Central"]
        assert len(out["violations"]) == 2  # Hilton (rate+stars), Marriott (rate+stars)
