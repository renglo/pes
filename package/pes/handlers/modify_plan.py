from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import importlib.resources
import json
import time
import uuid
import os
from decimal import Decimal
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

try:
    import yaml
except ImportError:
    yaml = None

from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController
from renglo.blueprint.blueprint_controller import BlueprintController

from contextvars import ContextVar


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

@dataclass
class RequestContext:
    portfolio: str = ''
    org: str = ''
    search_results: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    case_group: str = ''
    init: Dict[str, Any] = field(default_factory=dict)
    trip: Dict[str, Any] = field(default_factory=dict)
    
    
    

# ────────────────────────────────────────────────────────────────────────────────
# Core data shapes
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step_id: int  # Sequential index for ordering and dependencies (0, 1, 2...)
    title: str
    action: str
    inputs: Dict[str, Any]
    enter_guard: str = ""
    success_criteria: str = ""
    depends_on: List[int] = field(default_factory=list)  # References step_id
    next_step: Optional[int] = None  # Points to next step_id in execution order

@dataclass
class Plan:
    id: str
    steps: List[PlanStep]
    meta: Dict[str, Any] = field(default_factory=dict)

    
    

# ────────────────────────────────────────────────────────────────────────────────
# Actions + Action Catalog
# ────────────────────────────────────────────────────────────────────────────────    
@dataclass
class ActionSpec:
    key: str
    description: str
    required_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)
    success_criteria_hint: str = ""
# ────────────────────────────────────────────────────────────────────────────────
# Safe-ish bool eval
# ────────────────────────────────────────────────────────────────────────────────

def eval_bool(expr: str, context: Dict[str, Any]) -> bool:
    if not expr: return True
    safe_names = {"True": True, "False": False, "None": None, "len": len}
    try:
        return bool(eval(expr, {"__builtins__": {}}, {**safe_names, **context}))
    except Exception:
        return False



# ────────────────────────────────────────────────────────────────────────────────
# IntentModifier (internal to ModifyPlan)
# ────────────────────────────────────────────────────────────────────────────────


def _clamp_date_mod(d: Optional[str], now_date: str) -> Optional[str]:
    if not d or not isinstance(d, str) or len(d) != 10:
        return d
    return d if d >= now_date else now_date


def _parse_date_mod(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str) or len(s) < 10:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _intent_summary_for_delta(intent: Dict[str, Any]) -> Dict[str, Any]:
    iti = intent.get("itinerary") or {}
    lod = iti.get("lodging") or {}
    segs = iti.get("segments") or []
    stays = lod.get("stays") or []
    return {
        "party": intent.get("party") or {},
        "trip_type": iti.get("trip_type"),
        "converging": iti.get("converging"),
        "segments": [
            {
                "origin": (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin"),
                "destination": (s.get("destination") or {}).get("code") if isinstance(s.get("destination"), dict) else s.get("destination"),
                "depart_date": s.get("depart_date"),
            }
            for s in segs
        ],
        "stays": [{"location_code": s.get("location_code"), "check_in": s.get("check_in"), "check_out": s.get("check_out")} for s in stays],
        "lodging": {"check_in": lod.get("check_in"), "check_out": lod.get("check_out"), "number_of_nights": lod.get("number_of_nights")},
    }


class IntentModifier:
    """Extracts delta from user correction via LLM, applies programmatically to intent. Uses cases as canonical examples."""

    def __init__(self, agu, prompts: Optional[Dict[str, str]] = None):
        self.AGU = agu
        self.prompts = prompts or {}
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from pes.handlers.generate_plan import AIResponsesLLM
            self._llm = AIResponsesLLM(self.AGU)
        return self._llm

    def modify(self, base_intent: Dict[str, Any], user_correction: str,
               cases: Optional[List] = None) -> Optional[Dict[str, Any]]:
        delta = self._extract_delta(base_intent, user_correction, cases=cases)
        if not delta or not delta.get("changes"):
            return None
        intent = copy.deepcopy(base_intent)
        intent["request"] = intent.get("request") or {}
        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = ZoneInfo("America/New_York")
        now_date = datetime.now(tz).strftime("%Y-%m-%d")
        intent["request"]["now_date"] = now_date
        intent["request"]["user_message"] = user_correction
        for change in delta.get("changes", []):
            self._apply_change(intent, change, now_date)
        intent["updated_at"] = int(time.time())
        return intent

    def _extract_delta(self, base_intent: Dict[str, Any], user_correction: str,
                       cases: Optional[List] = None) -> Optional[Dict[str, Any]]:
        prompt_template = self.prompts.get("modify_intent_delta", "") or self._default_delta_prompt()
        existing = _intent_summary_for_delta(base_intent)
        modification_examples = self._build_modification_examples(cases)
        prompt = prompt_template.replace("#existing_intent#", json.dumps(existing, indent=2))
        prompt = prompt.replace("#user_correction#", user_correction)
        prompt = prompt.replace("#now_date#", datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"))
        prompt = prompt.replace("#modification_examples#", modification_examples)
        data = self._get_llm().complete_json(prompt)
        if not data or not isinstance(data.get("changes"), list):
            return None
        return data

    def _build_modification_examples(self, cases: Optional[List]) -> str:
        """Build intent structure examples from cases (from DB: {'intent', 'plan'} or VDBItem with .text)."""
        if not cases:
            return "[]"
        examples = []
        for c in cases[:3]:
            try:
                if isinstance(c, dict):
                    ex_intent = c.get("intent", c.get("trip_intent"))
                else:
                    text = getattr(c, 'text', str(c))
                    obj = json.loads(text) if isinstance(text, str) else text
                    ex_intent = obj.get("intent", obj.get("trip_intent"))
                if isinstance(ex_intent, str):
                    try:
                        ex_intent = json.loads(ex_intent)
                    except Exception:
                        continue
                if ex_intent and isinstance(ex_intent, dict):
                    examples.append(_intent_summary_for_delta(ex_intent))
            except Exception:
                pass
        return json.dumps(examples, indent=2) if examples else "[]"

    def _default_delta_prompt(self) -> str:
        return """TASK: MODIFY_INTENT_DELTA

You are a trip modification analyzer. Given the EXISTING INTENT and USER CORRECTION, output a structured delta describing ONLY what changed. Do NOT return the full intent.

Time context: Today is #now_date#. Use YYYY-MM-DD for dates.

CHANGE TYPES:
- extend_stay: User wants to stay longer. Use until_date (new check_out) or add_nights.
- shorten_stay: User wants to leave earlier. Use until_date or remove_nights.
- add_side_trip: "Add X nights in [City]", "stop in [City] before going home" = add city BEFORE return. Use city (IATA) and nights.
- change_dates: User changes departure or return. Use departure_date and/or return_date.

SIMILAR INTENTS (canonical examples):
#modification_examples#

EXISTING INTENT:
#existing_intent#

USER CORRECTION:
#user_correction#

Return ONLY valid JSON:
{
  "changes": [
    {"type": "extend_stay", "until_date": "YYYY-MM-DD"},
    {"type": "add_side_trip", "city": "LAS", "nights": 2},
    {"type": "change_dates", "departure_date": "YYYY-MM-DD", "return_date": "YYYY-MM-DD"}
  ]
}
"""

    def _apply_change(self, intent: Dict[str, Any], change: Dict[str, Any], now_date: str) -> None:
        ctype = change.get("type")
        if ctype == "extend_stay":
            self._apply_extend_stay(intent, change, now_date)
        elif ctype == "shorten_stay":
            self._apply_shorten_stay(intent, change, now_date)
        elif ctype == "add_side_trip":
            self._apply_add_side_trip(intent, change, now_date)
        elif ctype == "change_dates":
            self._apply_change_dates(intent, change, now_date)

    def _apply_extend_stay(self, intent: Dict[str, Any], change: Dict[str, Any], now_date: str) -> None:
        until = _clamp_date_mod(change.get("until_date"), now_date)
        add_nights = change.get("add_nights")
        lod = intent.setdefault("itinerary", {}).setdefault("lodging", {})
        stays = lod.get("stays") or []
        if until:
            for s in stays:
                s["check_out"] = until
            if lod.get("check_out"):
                lod["check_out"] = until
            if add_nights is not None:
                lod["number_of_nights"] = (lod.get("number_of_nights") or 0) + int(add_nights)
        elif add_nights is not None:
            add_n = int(add_nights)
            last_co = None
            for s in stays:
                co = _parse_date_mod(s.get("check_out"))
                if co and (last_co is None or co > last_co):
                    last_co = co
            if last_co:
                new_co = (last_co + timedelta(days=add_n)).strftime("%Y-%m-%d")
                new_co = _clamp_date_mod(new_co, now_date)
                for s in stays:
                    s["check_out"] = new_co
                if lod.get("check_out"):
                    lod["check_out"] = new_co
        ret_date = until or (stays[-1].get("check_out") if stays else None)
        if ret_date:
            self._update_return_segments_depart_date(intent, ret_date)

    def _apply_shorten_stay(self, intent: Dict[str, Any], change: Dict[str, Any], now_date: str) -> None:
        until = _clamp_date_mod(change.get("until_date"), now_date)
        lod = intent.setdefault("itinerary", {}).setdefault("lodging", {})
        stays = lod.get("stays") or []
        if until:
            for s in stays:
                s["check_out"] = until
            if lod.get("check_out"):
                lod["check_out"] = until
            self._update_return_segments_depart_date(intent, until)

    def _apply_add_side_trip(self, intent: Dict[str, Any], change: Dict[str, Any], now_date: str) -> None:
        city = (change.get("city") or "").upper()[:3]
        nights = int(change.get("nights") or 1)
        if not city:
            return
        iti = intent.setdefault("itinerary", {})
        segs = list(iti.get("segments") or [])
        lod = iti.setdefault("lodging", {})
        stays = list(lod.get("stays") or [])
        if not stays:
            return
        last_stay = stays[-1]
        depart_date = _clamp_date_mod(last_stay.get("check_out"), now_date)
        if not depart_date:
            return
        current_dest = last_stay.get("location_code")
        if isinstance(current_dest, dict):
            current_dest = current_dest.get("code")
        if not current_dest:
            return
        party = intent.get("party") or {}
        traveler_ids = party.get("traveler_ids") or []
        converging = iti.get("converging", False)
        extras = intent.get("extras") or {}
        converging_travelers = extras.get("travelers") or []
        seg_to_new_city = {
            "segment_id": f"seg_side_{len(segs)}",
            "origin": {"type": "airport", "code": current_dest},
            "destination": {"type": "airport", "code": city},
            "depart_date": depart_date,
            "passengers": sum(int(ct.get("count", 0) or 0) for ct in converging_travelers) if converging else len(traveler_ids) or 1,
            "transport_mode": "flight",
            "depart_time_window": {"start": None, "end": None},
        }
        if traveler_ids:
            seg_to_new_city["traveler_ids"] = traveler_ids
        segs.append(seg_to_new_city)
        ci_dt = _parse_date_mod(depart_date)
        if not ci_dt:
            return
        co_dt = ci_dt + timedelta(days=nights)
        check_out_new = _clamp_date_mod(co_dt.strftime("%Y-%m-%d"), now_date)
        new_stay = {"location_code": city, "check_in": depart_date, "check_out": check_out_new, "number_of_guests": seg_to_new_city["passengers"]}
        if traveler_ids:
            new_stay["traveler_ids"] = traveler_ids
        stays.append(new_stay)
        if converging and converging_travelers:
            for i, ct in enumerate(converging_travelers):
                orig = (ct.get("origin") or "").upper()
                if len(orig) == 3:
                    group_ids = self._traveler_ids_for_group(intent, i)
                    ret_seg = {"segment_id": f"seg_return_{len(segs)+i}", "origin": {"type": "airport", "code": city}, "destination": {"type": "airport", "code": orig}, "depart_date": check_out_new, "passengers": int(ct.get("count", 0) or 0), "transport_mode": "flight", "depart_time_window": {"start": None, "end": None}}
                    if group_ids:
                        ret_seg["traveler_ids"] = group_ids
                    segs.append(ret_seg)
            for s in [x for x in iti.get("segments") or [] if self._is_return_segment(x, current_dest)]:
                if s in segs:
                    segs.remove(s)
        else:
            origin_code = None
            for s in segs:
                d = s.get("destination")
                d_code = (d.get("code") if isinstance(d, dict) else d) if d else None
                if d_code == current_dest:
                    o = s.get("origin")
                    origin_code = (o.get("code") if isinstance(o, dict) else o) if o else None
                    break
            if origin_code:
                ret_seg = {"segment_id": "seg_return", "origin": {"type": "airport", "code": city}, "destination": {"type": "airport", "code": origin_code}, "depart_date": check_out_new, "passengers": seg_to_new_city["passengers"], "transport_mode": "flight", "depart_time_window": {"start": None, "end": None}}
                if traveler_ids:
                    ret_seg["traveler_ids"] = traveler_ids
                for x in [y for y in segs if self._is_return_segment(y, current_dest)]:
                    segs.remove(x)
                segs.append(ret_seg)
        iti["segments"] = segs
        lod["stays"] = stays

    def _is_return_segment(self, seg: Dict[str, Any], from_dest: str) -> bool:
        o = seg.get("origin")
        o_code = (o.get("code") if isinstance(o, dict) else o) if o else None
        return str(o_code or "").upper() == str(from_dest).upper()

    def _traveler_ids_for_group(self, intent: Dict[str, Any], group_index: int) -> List[str]:
        ct = (intent.get("extras") or {}).get("travelers") or []
        if group_index >= len(ct):
            return intent.get("party", {}).get("traveler_ids") or []
        idx = 0
        for i, c in enumerate(ct):
            n = int(c.get("count", 0) or 0)
            if i == group_index:
                return [f"t{j+1}" for j in range(idx, idx + n)]
            idx += n
        return []

    def _apply_change_dates(self, intent: Dict[str, Any], change: Dict[str, Any], now_date: str) -> None:
        dep = _clamp_date_mod(change.get("departure_date"), now_date)
        ret = _clamp_date_mod(change.get("return_date"), now_date)
        iti = intent.setdefault("itinerary", {})
        lod = iti.get("lodging", {})
        stays = lod.get("stays") or []
        if dep:
            segs = iti.get("segments") or []
            outbound = None
            for s in segs:
                d = s.get("destination")
                o = s.get("origin")
                d_code = (d.get("code") if isinstance(d, dict) else d) if d else None
                o_code = (o.get("code") if isinstance(o, dict) else o) if o else None
                if d_code and (not o_code or str(o_code).upper() != str(d_code).upper()):
                    outbound = s
                    break
            if not outbound and segs:
                outbound = segs[0]
            if outbound:
                outbound["depart_date"] = dep
            if stays:
                stays[0]["check_in"] = dep
            if lod.get("check_in"):
                lod["check_in"] = dep
        if ret:
            self._update_return_segments_depart_date(intent, ret)
            if stays:
                stays[-1]["check_out"] = ret
            if lod.get("check_out"):
                lod["check_out"] = ret

    def _update_return_segments_depart_date(self, intent: Dict[str, Any], new_date: str) -> None:
        iti = intent.get("itinerary") or {}
        segs = iti.get("segments") or []
        lod = iti.get("lodging", {})
        stays = lod.get("stays") or []
        dest_code = (stays[-1].get("location_code") if stays else None) or ((segs[0].get("destination") or {}).get("code") if segs else None)
        for s in segs:
            sid = str(s.get("segment_id", ""))
            o = s.get("origin")
            o_code = (o.get("code") if isinstance(o, dict) else o) if o else None
            if sid.startswith("seg_return") or "return" in sid or (dest_code and str(o_code or "").upper() == str(dest_code).upper()):
                s["depart_date"] = new_date


# ────────────────────────────────────────────────────────────────────────────────
# Handler implementation
# ────────────────────────────────────────────────────────────────────────────────

request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

'''
ModifyPlan: intent + modification request -> updated intent -> plan.
Step 1: Generate updated intent (internal IntentModifier).
Step 2: Propose plan (ProposePlan handler).
'''

class ModifyPlan:
    def __init__(self, prompts: Optional[Dict[str, str]] = None):
        # Load config for handlers (independent of Flask)
        self.config = load_config()
        
        self.AGU = None
        self.DAC = DataController(config=self.config)
        self.BPC = BlueprintController(config=self.config)
        
        # Store prompts passed during initialization (from text files or database)
        self.prompts = prompts or {}
        

    def _get_context(self) -> RequestContext:
        """Get the current request context."""
        return request_context.get()

    def _set_context(self, context: RequestContext):
        """Set the current request context."""
        request_context.set(context)

    def _update_context(self, **kwargs):
        """Update specific fields in the current request context."""
        context = self._get_context()
        for key, value in kwargs.items():
            setattr(context, key, value)
        self._set_context(context)
    
    def _load_prompts(self, portfolio: str, org: str, prompt_ring: str = "pes_prompts", case_group: str = None) -> Dict[str, str]:
        """
        Load prompts from database.
        Returns a dictionary with keys: 'to_intent', 'compose_plan_light', 'modify_plan'
        """
        prompts = {
            'to_intent': '',
            'modify_intent_delta': '',
            'compose_plan_light': '',
            'modify_plan': ''
        }
        
        try:
            # Get all prompt records for the case_group
            if not case_group:
                raise Exception('No case group')
            
            query = {
                'portfolio': portfolio,
                'org': org,
                'ring': prompt_ring,
                'value': case_group,
                'limit': 99,
                'operator': 'begins_with',
                'lastkey': None,
                'sort': 'asc'
            }
            response = self.DAC.get_a_b_query(query)
            
            #response = self.DAC.get_a_b(portfolio, org, prompt_ring, limit=1000)
            if response and 'items' in response:
                for item in response['items']:
                    # Get the key field (exact match from database)
                    key = item.get('key', '').lower()
                    
                    # Get prompt text from 'prompt' field only
                    prompt_text = item.get('prompt', '')
                    
                    # Strip leading whitespace (database responses have leading spaces)
                    if prompt_text:
                        prompt_text = prompt_text.lstrip()
                    
                    # Map keys to prompt types using exact match only
                    if key == 'to_intent':
                        prompts['to_intent'] = prompt_text
                    elif key == 'modify_intent_delta':
                        prompts['modify_intent_delta'] = prompt_text
                    elif key == 'compose_plan_light':
                        prompts['compose_plan_light'] = prompt_text
                    elif key == 'modify_plan':
                        prompts['modify_plan'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts from database: {str(e)}')
            
        
        return prompts

    def _load_prompts_from_package(self, package_route: str) -> Dict[str, str]:
        """
        Load prompts from a Python package using importlib.resources.
        Works regardless of filesystem location (installed package, zip, etc.).

        Args:
            package_route: Dotted path to the package subdirectory containing prompt YAML files.
                          Example: 'noma.prompts.pes' loads from noma/prompts/pes/*.yaml

        Returns:
            Dictionary with keys: 'to_intent', 'modify_intent_delta', 'compose_plan_light', 'modify_plan'
        """
        
        print('Using prompts from package')
        
        prompts = {
            'to_intent': '',
            'modify_intent_delta': '',
            'compose_plan_light': '',
            'modify_plan': ''
        }
        if not yaml:
            print('Warning: PyYAML not installed. Cannot load prompts from package.')
            return prompts
        try:
            parts = package_route.split('.')
            if not parts:
                raise ValueError('package_route must be non-empty (e.g. noma.prompts.pes)')
            package_name = parts[0]
            path_parts = parts[1:] if len(parts) > 1 else []
            resource_root = importlib.resources.files(package_name)
            for p in path_parts:
                resource_root = resource_root / p
            if not resource_root.is_dir():
                raise FileNotFoundError(f'Package resource path does not exist: {package_route}')
            for entry in resource_root.iterdir():
                if entry.name.endswith('.yaml') or entry.name.endswith('.yml'):
                    try:
                        with importlib.resources.as_file(entry) as f:
                            with open(f, 'r') as fp:
                                raw = yaml.safe_load(fp)
                        if not isinstance(raw, dict):
                            continue
                        key = (raw.get('key') or '').lower()
                        prompt_text = raw.get('prompt', '') or ''
                        if prompt_text:
                            prompt_text = prompt_text.lstrip()
                        if key in prompts:
                            prompts[key] = prompt_text
                    except Exception as e:
                        print(f'Warning: Could not load prompt from {entry.name}: {e}')
        except Exception as e:
            print(f'Warning: Could not load prompts from package {package_route}: {e}')
        return prompts

    def _load_seed_cases(self, portfolio: str, org: str, case_ring: str = "pes_cases", case_group: str = None) -> List[Dict[str, Any]]:
        """Load cases from database. Returns list of {'intent', 'plan'} dicts."""
        cases = []
        try:
            if not case_group:
                return cases
            query = {
                'portfolio': portfolio,
                'org': org,
                'ring': case_ring,
                'value': case_group,
                'limit': 99,
                'operator': 'begins_with',
                'lastkey': None,
                'sort': 'asc'
            }
            response = self.DAC.get_a_b_query(query)
            if response and 'items' in response:
                for item in response['items']:
                    intent_text = item.get('intent', item.get('trip_intent', ''))
                    plan_data = item.get('plan', {})
                    if intent_text and plan_data:
                        cases.append({'intent': intent_text, 'plan': plan_data})
        except Exception as e:
            print(f'Warning: Could not load cases from database: {e}')
        return cases
    
    def _load_actions(self, portfolio: str, org: str, action_ring: str = "schd_actions") -> List[ActionSpec]:
        """
        Load action catalog from database (schd_actions ring).
        Returns a list of ActionSpec objects.
        """
        actions = []
        
        try:
            # Get all action records from the ring
            response = self.DAC.get_a_b(portfolio, org, action_ring, limit=1000)
            if response and 'items' in response:
                for item in response['items']:
                    name = item.get('name', '')
                    key = item.get('key', '')
                    goal = item.get('goal', '')
                    tools_ref = item.get('tools_reference', '')
                    slots = item.get('slots', '')
                    
                    # Parse slots field for argument definitions
                    # Format expected: JSON string with 'required' and 'optional' arrays
                    required_args = []
                    optional_args = []
                    
                    if slots:
                        try:
                            # Try parsing as JSON first
                            if isinstance(slots, str):
                                # Try to parse as JSON
                                slots_data = json.loads(slots)
                            else:
                                slots_data = slots
                            
                            if isinstance(slots_data, dict):
                                required_args = slots_data.get('required', [])
                                optional_args = slots_data.get('optional', [])
                            elif isinstance(slots_data, list):
                                # If it's a list, assume all are required
                                required_args = slots_data
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, try parsing as comma-separated or newline-separated
                            if isinstance(slots, str):
                                # Try splitting by comma or newline
                                parts = [p.strip() for p in slots.replace('\n', ',').split(',') if p.strip()]
                                required_args = parts
                        except Exception as e:
                            print(f'Warning: Could not parse slots for action {name or key}: {str(e)}')
                    
                    # Use goal as description, or fallback to name
                    description = goal or f"Action: {name or key}"
                    
                    # Get success criteria from verification field if available
                    success_criteria_hint = item.get('verification', '')
                    
                    if key:
                        action_key = key
                        actions.append(ActionSpec(
                            key=action_key,
                            description=description,
                            required_args=required_args if required_args else [],
                            optional_args=optional_args if optional_args else [],
                            success_criteria_hint=success_criteria_hint
                        ))
        except Exception as e:
            print(f'Warning: Could not load actions from database: {str(e)}')
            
        
        return actions
    
    
    def run(self, payload):
        
        '''
        Payload
        {
            "portfolio":"",
            "org":"",
            "case_group":"",
            "message":"",  # Human-readable paragraph describing the modifications to make
            "plan": {},    # Existing plan to modify (Plan object with id, steps, meta)
            "state_machine": {},  # Optional: Execution state machine showing which steps are completed
            "trip": {}     # Optional: Trip document showing existing reservations
        }
        '''
        # Initialize a new request context
        function = 'run > modify_plan'
        context = RequestContext()
        
        if 'portfolio' in payload:
            context.portfolio = payload['portfolio']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No portfolio provided'}
        
        if 'org' in payload:
            context.org = payload['org']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No org provided'}
        
        if 'case_group' in payload:
            context.case_group = payload['case_group']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No case group provided'}
        
        if 'message' not in payload:
            return {'success':False,'function':function,'input':payload,'output':'No modification request (message) provided'}
        
        if 'plan' not in payload:
            return {'success':False,'function':function,'input':payload,'output':'No existing plan provided'}
        
        if '_init' in payload:
            raw = payload['_init']
            context.init = json.loads(raw) if isinstance(raw, str) else raw
        else:
            context.init = {}
            
            
        if 'trip' in payload:
            raw_trip = payload['trip']
            context.trip = json.loads(raw_trip) if isinstance(raw_trip, str) else raw_trip
        else:
            context.trip = {}


        

        try:
            self._set_context(context)
            
            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                'some_entity_type',
                'some_entity_id',
                'some_thread'
            )
            
            results = []
            print('Initializing PES>ModifyPlan')
            plan_actions = (context.init.get('plan_actions') or None) if isinstance(context.init, dict) else None
            
            base_intent = payload.get('intent')
            modification_request = payload['message']
            if not base_intent or not isinstance(base_intent, dict):
                return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_plan/run: No intent provided. Plan modification requires intent from generate_plan cache.'}
            try:
                prompt_route = (context.init or {}).get('prompt_route') if isinstance(context.init, dict) else None
                if prompt_route:
                    prompts = self._load_prompts_from_package(prompt_route)
                else:
                    prompts = self._load_prompts(context.portfolio, context.org, case_group=context.case_group)
                modifier = IntentModifier(agu=self.AGU, prompts=prompts)
                cases = self._load_seed_cases(
                    context.portfolio, context.org,
                    case_ring="pes_cases", case_group=context.case_group
                )[:3]
                updated_intent = modifier.modify(base_intent, modification_request, cases=cases)
                if not updated_intent:
                    return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_plan/run: Could not update intent from modification request.'}
                try:
                    from inca.handlers import Patcher
                    patcher = Patcher()
                    patcher.apply_invalidations_for_modification(base_intent, updated_intent)
                except ImportError:
                    pass  # inca optional: invalidate holds/bookings when intent changes
                from pes.handlers.propose_plan import ProposePlan
                propose_payload = {
                    'portfolio': context.portfolio,
                    'org': context.org,
                    'case_group': context.case_group,
                    'intent': updated_intent,
                    '_init': context.init,
                }
                response = ProposePlan().run(propose_payload)
                if response.get('success') and response.get('output'):
                    canonical = {
                        "plan": response["output"]["plan"],
                        "intent": updated_intent
                    }
                    return {'success': True, 'interface': 'plan', 'input': payload, 'output': canonical, 'stack': [response]}
                return {'success': False, 'function': function, 'input': payload, 'output': 'ERROR:@modify_plan/run: Could not generate plan from updated intent.'}
            except Exception as e:
                print(f'Intent-based regeneration failed: {e}')
                return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_plan/run: {str(e)}'}
            
        except Exception as e:
            print(f'Error during execution: {str(e)}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_plan/run: {str(e)}'}

