"""
Clear stale derived fields on a trip intent after it was modified out-of-band.

Clears stale quote/hold/booking selections on the revised intent when upstream itinerary or
constraints change. Operates on ``intent.working_memory`` in place.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _compute_changed_paths(before: Any, after: Any, prefix: str = "") -> List[str]:
    if type(before) != type(after):
        return [prefix or "$"]

    if isinstance(before, dict):
        keys = set(before.keys()) | set(after.keys())
        out: List[str] = []
        for k in keys:
            p = f"{prefix}.{k}" if prefix else k
            if k not in before or k not in after:
                out.append(p)
            else:
                out.extend(_compute_changed_paths(before[k], after[k], p))
        return out

    if isinstance(before, list):
        return [prefix or "$"] if before != after else []

    return [prefix or "$"] if before != after else []


def _ensure_working_memory_defaults(wm: Dict[str, Any]) -> None:
    wm.setdefault("flight_quotes", [])
    wm.setdefault("hotel_quotes", [])
    wm.setdefault("flight_quotes_by_segment", [])
    wm.setdefault("hotel_quotes_by_stay", [])
    wm.setdefault("ranked_bundles", [])
    wm.setdefault("risk_report", None)
    wm.setdefault("holds", [])
    wm.setdefault("bookings", [])
    sel = wm.setdefault(
        "selected",
        {
            "bundle_id": None,
            "flight_option_id": None,
            "hotel_option_id": None,
            "flight_option_ids": [],
            "hotel_option_ids": [],
        },
    )
    if isinstance(sel, dict):
        sel.setdefault("flight_option_ids", [])
        sel.setdefault("hotel_option_ids", [])


def _invalidate_working_memory_caches(
    intent: Dict[str, Any], changed_paths: List[str]
) -> Tuple[List[str], List[str]]:
    wm = intent.setdefault("working_memory", {})
    _ensure_working_memory_defaults(wm)

    cleared: List[str] = []
    reasons: List[str] = []

    def any_startswith(prefixes: List[str]) -> bool:
        return any(any(cp.startswith(pref) for pref in prefixes) for cp in changed_paths)

    def clear_key(key: str, reason: str) -> None:
        if key not in wm:
            return
        wm[key] = [] if isinstance(wm[key], list) else None
        cleared.append(f"working_memory.{key}")
        reasons.append(reason)

    def clear_selection(reason: str) -> None:
        wm["selected"] = {
            "bundle_id": None,
            "flight_option_id": None,
            "hotel_option_id": None,
            "flight_option_ids": [],
            "hotel_option_ids": [],
        }
        cleared.append("working_memory.selected")
        reasons.append(reason)

    if any_startswith(["itinerary.segments", "preferences.flight", "party.travelers"]):
        clear_key("flight_quotes", "Flight inputs changed → cleared flight quotes.")
        clear_key("flight_quotes_by_segment", "Flight inputs changed → cleared flight quotes by segment.")
        clear_key("ranked_bundles", "Flight inputs changed → cleared ranked bundles.")
        clear_key("risk_report", "Flight inputs changed → cleared risk report.")
        clear_key("holds", "Flight inputs changed → cleared holds.")
        clear_key("bookings", "Flight inputs changed → cleared bookings (intent no longer matches).")
        clear_selection("Selection cleared because flight-derived artifacts are stale.")

    if any_startswith(["itinerary.lodging", "preferences.hotel"]):
        clear_key("hotel_quotes", "Hotel inputs changed → cleared hotel quotes.")
        clear_key("hotel_quotes_by_stay", "Hotel inputs changed → cleared hotel quotes by stay.")
        clear_key("ranked_bundles", "Hotel inputs changed → cleared ranked bundles.")
        clear_key("risk_report", "Hotel inputs changed → cleared risk report.")
        clear_key("holds", "Hotel inputs changed → cleared holds.")
        clear_key("bookings", "Hotel inputs changed → cleared bookings (intent no longer matches).")
        clear_selection("Selection cleared because hotel-derived artifacts are stale.")

    if any_startswith(["policy", "constraints"]):
        if wm.get("risk_report") is not None:
            clear_key("risk_report", "Policy/constraints changed → cleared risk report.")

        if any(
            cp.startswith("constraints.budget_total")
            or cp.startswith("constraints.refundable_preference")
            for cp in changed_paths
        ):
            if wm.get("holds"):
                clear_key("holds", "Budget/refundability changed → cleared holds for safety.")

    return cleared, reasons


def apply_working_memory_invalidations_for_intent_modification(
    intent_before: Dict[str, Any],
    intent_after: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Diff ``intent_before`` vs ``intent_after``, then clear derived ``working_memory`` fields
    on ``intent_after`` when upstream inputs changed. Mutates ``intent_after`` in place.

    Returns ``(cleared_paths, reasons)`` for logging; safe to ignore.
    """
    if not isinstance(intent_before, dict) or not isinstance(intent_after, dict):
        return [], []
    changed = _compute_changed_paths(intent_before, intent_after)
    return _invalidate_working_memory_caches(intent_after, changed)
