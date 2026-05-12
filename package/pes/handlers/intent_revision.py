"""
Single-pass trip intent revision.

This module is intentionally independent of ``modify_intent`` (which calls it).
It implements the flow you described:

1. Take the current intent as authoritative state.
2. Apply fixed natural-language rules plus a small set of canonical intent examples
   (from ``pes_cases``) that illustrate the same rules.
3. Ask the model to return a *complete* replacement intent object (not a delta).

Downstream plan diffing belongs to another stage; this class only targets intent quality.

Customization (see ``resolve_intent_revision_rules``):

- **Primary rules** (replace the built-in default when set): ``_init.intent_revision_rules``,
  ``_init.intent_revision_rules_path``, prompt key ``intent_revision_rules`` (DB or package YAML),
  ``INTENT_REVISION_RULES`` (inline, or a filesystem path if that path exists as a file),
  ``INTENT_REVISION_RULES_PATH``.
- **Supplemental rules** (appended under "ADDITIONAL RULES"): prompt key ``modify_intent``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController


_DEFAULT_RULES = """\
RULES (follow exactly; examples below illustrate the same rules and must not contradict them):

1) OUTPUT SHAPE
   - Respond with JSON only: {\"intent\": <full updated intent object>}.
   - The value of \"intent\" must be a complete object: same top-level structure and conventions as CURRENT_INTENT (do not return patches, diffs, or prose).

2) PRESERVE UNLESS THE USER CHANGES IT
   - Keep every part of the trip the user did not ask to change (party, ids, segments they did not mention, etc.).
   - Do not delete or rewrite segments or stays unless the user’s request requires it.

3) DATES
   - Use ISO dates YYYY-MM-DD everywhere.
   - Hotel stay: check_out is exclusive-day style consistent with CURRENT_INTENT (nights = check_out - check_in in days).
   - When the user asks for \"N nights\" in a city, set the stay so that length matches N (fix both check_in and check_out, and any flight that should arrive that day).

4) SEGMENTS (FLIGHTS / TRANSPORT)
   - Every time the traveler moves to a new city for a stay, there must be a coherent transport segment before that stay unless the user explicitly wants surface transport omitted.
   - Keep segment order logical in time: outbound, then any multi-city legs, then return.
   - Align each segment’s depart_date with the stay it follows or precedes as appropriate (no overnight gaps unless the user asked).

5) LODGING
   - ``lodging.stays`` entries must match the verbal request (city, nights, order relative to segments).
   - If the trip has a single main destination and the user adds another city \"after\" it, insert a new stay and the necessary segment(s); shorten the previous city’s stay if the user implied leaving earlier.

6) PARTY / IDS
   - Preserve traveler_ids and passenger counts unless the user changes party size.

7) CONSISTENCY
   - After edits, re-read the whole intent and fix internal inconsistencies (duplicate dates, same check-in/check-out when nights > 0, missing return, etc.).
"""


def _truncate_json(obj: Any, max_chars: int) -> str:
    raw = json.dumps(obj, indent=2, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n... [truncated]"


def _read_text_file(path: str) -> Optional[str]:
    p = (path or "").strip()
    if not p:
        return None
    try:
        with open(p, encoding="utf-8") as f:
            t = f.read().strip()
        return t or None
    except OSError as ex:
        print(f"IntentRevisor: could not read rules file {p!r}: {ex}")
        return None


def resolve_intent_revision_rules(
    *,
    config: Dict[str, Any],
    prompts: Dict[str, Any],
    init: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the main RULES block and optional supplemental text for ``modify_intent``.

    Returns ``(rules_text, extra_rules)``:

    - ``rules_text``: non-empty means pass ``IntentRevisor(..., rules_text=...)`` so this
      text **replaces** the library default. ``None`` means use the built-in default rules.
    - ``extra_rules``: optional; from the ``modify_intent`` prompt, appended under
      "ADDITIONAL RULES FROM CALLER" in the user prompt.

    Precedence for ``rules_text`` (first non-empty wins):

    1. ``init["intent_revision_rules"]`` — inline string
    2. file at ``init["intent_revision_rules_path"]``
    3. ``prompts["intent_revision_rules"]`` — DB or package YAML (``key: intent_revision_rules``)
    4. ``config["INTENT_REVISION_RULES"]`` — if the trimmed value is an existing file path,
       that file's contents; otherwise the value is used as inline rules text
    5. file at ``config["INTENT_REVISION_RULES_PATH"]``
    """
    init = init if isinstance(init, dict) else {}

    rules: Optional[str] = None

    inline = (init.get("intent_revision_rules") or "").strip()
    if inline:
        rules = inline
    else:
        ipath = (init.get("intent_revision_rules_path") or "").strip()
        if ipath:
            rules = _read_text_file(ipath)
        if not rules:
            pr = (prompts.get("intent_revision_rules") or "").strip()
            if pr:
                rules = pr
        if not rules:
            cr = (config.get("INTENT_REVISION_RULES") or "").strip()
            if cr:
                rules = _read_text_file(cr) if os.path.isfile(cr) else cr
        if not rules:
            cpath = (config.get("INTENT_REVISION_RULES_PATH") or "").strip()
            if cpath:
                rules = _read_text_file(cpath)

    rules_out = (rules or "").strip() or None
    extra = (prompts.get("modify_intent") or "").strip() or None
    return (rules_out, extra)


class IntentRevisor:
    """
    Revise a trip ``intent`` in one LLM call using rules + optional canonical examples.

    Examples are loaded from the ``pes_cases`` ring (same query shape as planning seeds):
    ``begins_with`` on ``case_group``, up to ``max_examples`` rows that include both
    intent and plan payloads; only the **intent** side is shown to the model so rules
    and examples stay aligned (examples are real intents, not plan steps).
    """

    def __init__(
        self,
        agu: AgentUtilities,
        *,
        rules_text: str = _DEFAULT_RULES,
        max_examples: int = 3,
        per_example_char_cap: int = 3500,
        model: Optional[str] = None,
    ) -> None:
        self._agu = agu
        self._rules = (rules_text or "").strip() or _DEFAULT_RULES
        self._max_examples = max(0, int(max_examples))
        self._per_example_cap = max(500, int(per_example_char_cap))
        cfg = load_config()
        self._model = (model or cfg.get("INTENT_REVISION_MODEL") or getattr(agu, "AI_2_MODEL", None) or "gpt-4o-mini")

    def load_canonical_intent_examples(
        self,
        portfolio: str,
        org: str,
        case_group: Optional[str],
        *,
        case_ring: str = "pes_cases",
    ) -> List[Dict[str, Any]]:
        if not case_group or not str(case_group).strip():
            return []
        dac = DataController(config=load_config())
        query = {
            "portfolio": portfolio,
            "org": org,
            "ring": case_ring,
            "value": case_group,
            "limit": 99,
            "operator": "begins_with",
            "lastkey": None,
            "sort": "asc",
        }
        out: List[Dict[str, Any]] = []
        try:
            response = dac.get_a_b_query(query)
            for item in (response or {}).get("items") or []:
                raw = item.get("intent", item.get("trip_intent"))
                plan = item.get("plan")
                if not raw or not plan:
                    continue
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(raw, dict):
                    parsed = raw
                else:
                    continue
                if isinstance(parsed, dict) and parsed:
                    out.append(parsed)
                if len(out) >= self._max_examples:
                    break
        except Exception as ex:
            print(f"IntentRevisor: could not load pes_cases examples: {ex}")
        return out

    def build_user_prompt(
        self,
        *,
        current_intent: Dict[str, Any],
        user_message: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        extra_rules: Optional[str] = None,
    ) -> str:
        rules = self._rules
        if extra_rules and str(extra_rules).strip():
            rules = rules.strip() + "\n\nADDITIONAL RULES FROM CALLER:\n" + str(extra_rules).strip()

        blocks: List[str] = [
            "TASK: REVISE_TRIP_INTENT_FULL",
            "",
            "=== RULES ===",
            rules,
            "",
            "=== CANONICAL INTENT EXAMPLES (structure only; same rules as above) ===",
        ]
        ex_list = examples or []
        if not ex_list:
            blocks.append("(no examples loaded — follow RULES and CURRENT_INTENT structure strictly.)")
        else:
            for i, ex in enumerate(ex_list, start=1):
                blocks.append(f"--- example {i} ---")
                blocks.append(_truncate_json(ex, self._per_example_cap))

        blocks.extend(
            [
                "",
                "=== CURRENT_INTENT (authoritative starting point) ===",
                _truncate_json(current_intent, 12000),
                "",
                "=== USER REQUEST ===",
                str(user_message or "").strip(),
                "",
                "Return JSON: {\"intent\": <full updated intent object>}",
            ]
        )
        return "\n".join(blocks)

    def revise(
        self,
        *,
        portfolio: str,
        org: str,
        case_group: Optional[str],
        current_intent: Dict[str, Any],
        user_message: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        extra_rules: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Produce a full replacement intent dict, or ``None`` on failure.
        """
        if not isinstance(current_intent, dict) or not current_intent:
            return None
        msg = (user_message or "").strip()
        if not msg:
            return None

        ex = examples
        if ex is None:
            ex = self.load_canonical_intent_examples(portfolio, org, case_group)

        prompt = self.build_user_prompt(
            current_intent=current_intent,
            user_message=msg,
            examples=ex,
            extra_rules=extra_rules,
        )

        try:
            resp = self._agu.llm(
                {
                    "model": self._model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                }
            )
        except Exception as ex:
            print(f"IntentRevisor: llm call failed: {ex}")
            return None

        if resp is False or resp is None:
            return None

        text = getattr(resp, "content", None) or ""
        if not str(text).strip():
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        intent = data.get("intent") if isinstance(data, dict) else None
        if not isinstance(intent, dict) or not intent:
            return None
        return intent
