#!/usr/bin/env python3
"""
Build seed_cases.json from seed_cases_readable.json.

The readable format uses full intent structure (renglo.intent.v1).
This script runs intent_to_text() on each intent and produces
the compact format stored in the VDB.
"""
import json
import os
from typing import Any, Dict


def intent_to_text(ti: Dict[str, Any]) -> str:
    """Compact JSON for retrieval and plan generation. Matches generate_plan.intent_to_text."""
    out: Dict[str, Any] = {}
    party = ti.get("party") or {}
    travelers = party.get("travelers") or {}
    if travelers:
        out["party"] = {"travelers": travelers}
    iti = ti.get("itinerary") or {}
    if iti.get("trip_type"):
        out["trip_type"] = iti["trip_type"]
    segs = iti.get("segments") or []
    if segs:
        out["segments"] = [
            {
                "origin": (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin"),
                "destination": (s.get("destination") or {}).get("code") if isinstance(s.get("destination"), dict) else s.get("destination"),
                "depart_date": s.get("depart_date"),
            }
            for s in segs
        ]
    lod = iti.get("lodging") or {}
    if lod.get("stays"):
        out["stays"] = lod["stays"]
    elif lod.get("check_in") or lod.get("check_out"):
        out["lodging"] = {"check_in": lod.get("check_in"), "check_out": lod.get("check_out"), "location_hint": lod.get("location_hint")}
    prefs = ti.get("preferences") or {}
    if prefs:
        out["preferences"] = prefs
    constraints = ti.get("constraints") or {}
    if constraints and any(v for v in constraints.values() if v):
        out["constraints"] = constraints
    return json.dumps(out, sort_keys=True, separators=(",", ":"))

def main():
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(_script_dir)  # pes package dir (parent of scripts)
    readable_path = os.path.join(pkg_dir, "seed_cases_readable.json")
    output_path = os.path.join(pkg_dir, "seed_cases.json")

    with open(readable_path, "r") as f:
        cases = json.load(f)

    out = []
    for case in cases:
        intent = case.get("intent", {})
        plan = case.get("plan", {})
        meta = case.get("meta", {})
        case_id = case.get("id", "")

        intent_compact = intent_to_text(intent)
        text_obj = {"intent": intent_compact, "plan": plan}
        text_str = json.dumps(text_obj, separators=(",", ":"))

        out.append({
            "id": case_id,
            "text": text_str,
            "meta": meta
        })

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(out)} cases to {output_path}")

if __name__ == "__main__":
    main()
