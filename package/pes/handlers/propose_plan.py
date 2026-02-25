"""
ProposePlan: Standalone handler for intent → plan.
Takes intent (and optional cases) as input, returns a plan.
Called by GeneratePlan and ModifyPlan.
Contains all plan composition logic (compose_from_cases, compose_plan_light).
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json
import uuid
from contextvars import ContextVar
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController

from pes.handlers.generate_plan import (
    ActionSpec,
    Plan,
    PlanStep,
    VDBItem,
    intent_for_plan,
    DecimalEncoder,
    AIResponsesLLM,
)


@dataclass
class RequestContext:
    portfolio: str = ''
    org: str = ''
    case_group: str = ''
    init: Dict[str, Any] = field(default_factory=dict)


request_context: ContextVar = ContextVar('propose_plan_context', default=None)


def _replace_tokens(prompt: str, replacements: Dict[str, Any]) -> str:
    """Replace #token# placeholders in prompt template."""
    json_tokens = {
        'intent_text', 'example_cases', 'fact_texts', 'catalog_summary',
        'catalog', 'skills', 'plan_examples', 'activities_requested', 'plan_details', 'intent_examples'
    }
    for token, value in replacements.items():
        token_placeholder = '#' + token + '#'
        if token_placeholder in prompt:
            replacement = f'```json\n{value}\n```' if token in json_tokens else str(value)
            prompt = prompt.replace(token_placeholder, replacement)
        for legacy in ('{' + token + '}', '{{' + token + '}}'):
            if legacy in prompt:
                replacement = f'```json\n{value}\n```' if token in json_tokens else str(value)
                prompt = prompt.replace(legacy, replacement)
    return prompt


class ProposePlan:
    """
    Standalone handler: intent in, plan out.
    Contains all plan composition logic. No dependency on IntentGenerator for composition.
    Payload: portfolio, org, case_group, intent, _init (optional), cases (optional)
    Output: { success, output: { plan, intent } }
    """

    def __init__(self):
        self.config = load_config()
        self.DAC = DataController(config=self.config)
        self.AGU = None

    def _get_context(self):
        return request_context.get()

    def _set_context(self, ctx):
        request_context.set(ctx)

    def _load_prompts(self, portfolio: str, org: str, prompt_ring: str = "pes_prompts", case_group: str = None) -> Dict[str, str]:
        prompts = {'compose_plan': ''}
        try:
            if case_group:
                query = {
                    'portfolio': portfolio, 'org': org, 'ring': prompt_ring,
                    'value': case_group, 'limit': 99, 'operator': 'begins_with',
                    'lastkey': None, 'sort': 'asc'
                }
                response = self.DAC.get_a_b_query(query)
                if response and 'items' in response:
                    for item in response['items']:
                        key = item.get('key', '').lower()
                        prompt_text = (item.get('prompt') or '').lstrip()
                        if key == 'compose_plan' and prompt_text:
                            prompts['compose_plan'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts from database: {e}')
        if not prompts.get('compose_plan'):
            try:
                import yaml
                import importlib.resources
                resource_root = importlib.resources.files('noma') / 'prompts' / 'pes'
                for entry in resource_root.iterdir():
                    if entry.name.endswith(('.yaml', '.yml')) and 'compose_plan' in entry.name:
                        with importlib.resources.as_file(entry) as f:
                            with open(f, 'r') as fp:
                                raw = yaml.safe_load(fp)
                        if isinstance(raw, dict) and (raw.get('key') or '').lower() == 'compose_plan':
                            prompts['compose_plan'] = (raw.get('prompt') or '').lstrip()
                            break
            except Exception as e:
                print(f'Warning: Could not load compose_plan from package: {e}')
        return prompts

    def _load_actions(self, portfolio: str, org: str, action_ring: str = "schd_actions") -> List:
        actions = []
        try:
            response = self.DAC.get_a_b(portfolio, org, action_ring, limit=1000)
            if response and 'items' in response:
                for item in response['items']:
                    key = item.get('key', '')
                    slots = item.get('slots', '')
                    required_args, optional_args = [], []
                    if slots:
                        try:
                            slots_data = json.loads(slots) if isinstance(slots, str) else slots
                            if isinstance(slots_data, dict):
                                required_args = slots_data.get('required', [])
                                optional_args = slots_data.get('optional', [])
                        except Exception:
                            pass
                    if key:
                        actions.append(ActionSpec(
                            key=key,
                            description=item.get('goal', ''),
                            required_args=required_args,
                            optional_args=optional_args,
                            success_criteria_hint=item.get('verification', '')
                        ))
        except Exception as e:
            print(f'Warning: Could not load actions: {e}')
        return actions

    def _validate_and_patch_plan(self, plan: Plan, action_catalog: List[ActionSpec]) -> Plan:
        print(f'Validating plan with {len(plan.steps)} steps...')
        known = {t.key: t for t in action_catalog}
        validated: List[PlanStep] = []
        for s in plan.steps:
            if not s.action or s.action == "":
                continue
            if s.action not in known:
                continue
            spec = known[s.action]
            inputs = s.inputs or {}
            missing = [a for a in spec.required_args if a not in inputs]
            if missing:
                continue
            if not s.success_criteria and spec.success_criteria_hint:
                s.success_criteria = spec.success_criteria_hint
            if not s.enter_guard:
                s.enter_guard = "True"
            validated.append(s)
        plan.steps = validated
        return plan

    def _inject_traveler_ids_from_intent(self, plan: Plan, intent: Dict[str, Any]) -> Plan:
        """Inject traveler_ids from intent into plan steps when missing. Uses segment.traveler_ids for flights, stay.traveler_ids for hotels."""
        iti = intent.get("itinerary") or {}
        lod = iti.get("lodging") or {}
        segs = iti.get("segments") or []
        stays = lod.get("stays") or []
        party_tids = (intent.get("party") or {}).get("traveler_ids") or []

        seg_idx = 0
        stay_idx = 0
        for step in plan.steps:
            inputs = step.inputs or {}
            if inputs.get("traveler_ids"):
                continue
            if step.action == "quote_flight" and seg_idx < len(segs):
                seg = segs[seg_idx]
                tids = seg.get("traveler_ids") or party_tids
                if tids:
                    inputs["traveler_ids"] = tids
                seg_idx += 1
            elif step.action == "quote_hotel" and stay_idx < len(stays):
                stay = stays[stay_idx]
                tids = stay.get("traveler_ids") or party_tids
                if tids:
                    inputs["traveler_ids"] = tids
                stay_idx += 1
            elif party_tids:
                inputs["traveler_ids"] = party_tids
        return plan

    def _build_plan_from_intent(self, intent: Dict[str, Any], plan_id: str, action_catalog: List[ActionSpec]) -> Plan:
        """Build plan programmatically from intent."""
        def _code(v) -> str:
            if isinstance(v, dict):
                return v.get("code") or ""
            return str(v) if v else ""

        iti = intent.get("itinerary") or {}
        segs = iti.get("segments") or []
        lod = iti.get("lodging") or {}
        stays_list = lod.get("stays") or []
        party = intent.get("party") or {}
        travelers = party.get("travelers") or {}
        default_pax = int(travelers.get("adults", 0) or 0) + int(travelers.get("children", 0) or 0) + int(travelers.get("infants", 0) or 0) or 1

        dest_code = _code(segs[0].get("destination")) if segs else None
        last_dest = (stays_list[-1].get("location_code") if stays_list else None) or dest_code
        if isinstance(last_dest, dict):
            last_dest = last_dest.get("code") if last_dest else None
        last_dest = str(last_dest or "").upper() or (str(dest_code or "").upper() if dest_code else "")

        inbound: List[Dict] = []
        return_segs: List[Dict] = []
        intermediate: List[Dict] = []
        for seg in segs:
            d = _code(seg.get("destination"))
            o = _code(seg.get("origin"))
            if dest_code and d and str(d).upper() == str(dest_code).upper():
                inbound.append(seg)
            elif last_dest and o and str(o).upper() == str(last_dest).upper():
                return_segs.append(seg)
            else:
                intermediate.append(seg)
        if not inbound and not return_segs and segs:
            inbound = list(segs)

        steps: List[PlanStep] = []
        leg = 0

        for seg in inbound:
            o_code = _code(seg.get("origin"))
            d_code = _code(seg.get("destination"))
            dep = seg.get("depart_date")
            pax = seg.get("passengers", default_pax)
            tids = seg.get("traveler_ids") or []
            if o_code and d_code and dep:
                steps.append(PlanStep(
                    step_id=len(steps),
                    title=f"{o_code} to {d_code} flight",
                    action="quote_flight",
                    inputs={
                        "from_airport_code": o_code,
                        "to_airport_code": d_code,
                        "departure_date": dep,
                        "leg": leg,
                        "passengers": pax,
                        "traveler_ids": tids,
                    },
                    enter_guard="True",
                    success_criteria="len(result) > 0",
                    depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                    next_step=None,
                ))
                leg += 1

        prev_stay_loc = None
        for stay in stays_list:
            loc = stay.get("location_code") or dest_code
            loc_code = (_code(loc) if isinstance(loc, dict) else str(loc or "").upper()) or (str(dest_code or "").upper() if dest_code else "")
            ci = stay.get("check_in")
            co = stay.get("check_out")
            n_guests = stay.get("number_of_guests", default_pax)
            tids = stay.get("traveler_ids") or []
            if not loc or not ci or not co:
                continue
            if prev_stay_loc and loc_code and str(loc_code).upper() != str(prev_stay_loc).upper():
                for seg in intermediate:
                    o = _code(seg.get("origin"))
                    d = _code(seg.get("destination"))
                    if str(o).upper() == str(prev_stay_loc).upper() and str(d).upper() == str(loc_code).upper():
                        dep = seg.get("depart_date")
                        pax = seg.get("passengers", default_pax)
                        seg_tids = seg.get("traveler_ids") or []
                        if o and d and dep:
                            steps.append(PlanStep(
                                step_id=len(steps),
                                title=f"{o} to {d} flight",
                                action="quote_flight",
                                inputs={
                                    "from_airport_code": o,
                                    "to_airport_code": d,
                                    "departure_date": dep,
                                    "leg": leg,
                                    "passengers": pax,
                                    "traveler_ids": seg_tids,
                                },
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                                next_step=None,
                            ))
                            leg += 1
                        break
            prev_stay_loc = loc_code
            try:
                from datetime import datetime
                ci_dt = datetime.strptime(str(ci)[:10], "%Y-%m-%d")
                co_dt = datetime.strptime(str(co)[:10], "%Y-%m-%d")
                nights = max(1, (co_dt - ci_dt).days)
            except Exception:
                nights = 1
            n_guests_str = str(n_guests) if isinstance(n_guests, (int, float)) else n_guests
            steps.append(PlanStep(
                step_id=len(steps),
                title=f"{loc} hotel {nights} nights ({n_guests_str} guests)",
                action="quote_hotel",
                inputs={
                    "city": loc,
                    "area": None,
                    "check_in_date": ci,
                    "number_of_nights": str(nights),
                    "number_of_guests": n_guests_str,
                    "traveler_ids": tids,
                },
                enter_guard="True",
                success_criteria="len(result) > 0",
                depends_on=[len(steps) - 1] if steps else [],
                next_step=None,
            ))

        for seg in return_segs:
            o_code = _code(seg.get("origin"))
            d_code = _code(seg.get("destination"))
            dep = seg.get("depart_date")
            pax = seg.get("passengers", default_pax)
            tids = seg.get("traveler_ids") or []
            if o_code and d_code and dep:
                steps.append(PlanStep(
                    step_id=len(steps),
                    title=f"{o_code} to {d_code} flight",
                    action="quote_flight",
                    inputs={
                        "from_airport_code": o_code,
                        "to_airport_code": d_code,
                        "departure_date": dep,
                        "leg": leg,
                        "passengers": pax,
                        "traveler_ids": tids,
                    },
                    enter_guard="True",
                    success_criteria="len(result) > 0",
                    depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                    next_step=None,
                ))
                leg += 1

        for i, step in enumerate(steps):
            step.step_id = i
            step.next_step = None if i == len(steps) - 1 else i + 1
            step.depends_on = [i - 1] if i > 0 else []

        return Plan(id=plan_id, steps=steps, meta={"strategy": "programmatic"})

    def compose_plan_light(self, intent: Dict[str, Any], action_catalog: List[ActionSpec]) -> Plan:
        """Build plan from intent programmatically."""
        plan_id = uuid.uuid4().hex[:8]
        plan = self._build_plan_from_intent(intent, plan_id, action_catalog)
        return self._validate_and_patch_plan(plan, action_catalog)

    def compose_from_cases(self, intent: Dict[str, Any], llm: AIResponsesLLM, action_catalog: List[ActionSpec],
                          prompts: Dict[str, str], cases: Optional[List[VDBItem]] = None) -> Plan:
        """Compose plan from intent using LLM. Cases provide canonical intent→plan examples."""
        print('Composing plan from intent (with cases as examples)...')
        catalog = [{
            "name": t.key,
            "description": t.description,
            "required_args": t.required_args,
            "optional_args": t.optional_args,
            "success_criteria_hint": t.success_criteria_hint
        } for t in action_catalog]
        intent_json = json.dumps(intent_for_plan(intent), indent=2)

        plan_examples = "[]"
        if cases:
            examples = []
            for c in cases[:3]:
                try:
                    obj = json.loads(c.text)
                    ex_intent = obj.get("intent", obj.get("trip_intent"))
                    ex_plan = obj.get("plan", {})
                    if isinstance(ex_intent, str):
                        try:
                            ex_intent = json.loads(ex_intent)
                        except Exception:
                            continue
                    steps = ex_plan.get("steps", []) if isinstance(ex_plan, dict) else []
                    if ex_intent and steps:
                        intent_plan = intent_for_plan(ex_intent) if isinstance(ex_intent, dict) else ex_intent
                        step_dicts = [s if isinstance(s, dict) else asdict(s) for s in steps[:10]]
                        examples.append({"intent": intent_plan, "plan": {"steps": step_dicts}})
                except Exception:
                    pass
            if examples:
                plan_examples = json.dumps(examples, indent=2, cls=DecimalEncoder)

        prompt_template = prompts.get('compose_plan', '')
        plan_id = uuid.uuid4().hex[:8]
        if prompt_template:
            replacements = {
                'intent_text': intent_json,
                'sig_text': intent_json,
                'catalog': json.dumps(catalog, indent=2),
                'plan_examples': plan_examples,
                'plan_id': plan_id
            }
            prompt = _replace_tokens(prompt_template, replacements)
        else:
            prompt = f"""You are a planner that composes a COMPLETE executable plan using ONLY the allowed actions.
TASK: COMPOSE_PLAN
CANONICAL EXAMPLES (intent → plan): {plan_examples}
Intent: ```json\n{intent_json}\n```
ACTION_CATALOG: ```json\n{json.dumps(catalog, indent=2)}\n```
Return ONLY JSON: {{"plan": {{"id":"{plan_id}", "meta": {{"strategy":"compose"}}, "steps": [PlanStep..]}}}}
Each PlanStep: {{"step_id": 0, "title": "...", "action": "<from catalog>", "inputs": {{...}}, "enter_guard": "True", "success_criteria": "...", "depends_on": [], "next_step": 1}}
"""
        data = llm.complete_json(prompt)
        if not data or "plan" not in data:
            print('[ERROR] LLM returned invalid plan structure')
            return Plan(id=plan_id, steps=[], meta={"strategy": "compose", "error": "LLM returned invalid plan"})

        steps = []
        for idx, s in enumerate(data["plan"]["steps"]):
            if "step_id" not in s:
                s["step_id"] = len(steps)
            if "depends_on" not in s:
                s["depends_on"] = []
            if "next_step" not in s:
                s["next_step"] = None
            try:
                steps.append(PlanStep(**s))
            except Exception as e:
                print(f'[ERROR] Failed to create PlanStep from step {idx}: {e}')
                continue
        plan = Plan(id=data["plan"]["id"], steps=steps, meta=data["plan"].get("meta", {}))
        plan = self._inject_traveler_ids_from_intent(plan, intent)
        return self._validate_and_patch_plan(plan, action_catalog)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        function = 'run > propose_plan'
        if not payload.get('portfolio'):
            return {'success': False, 'function': function, 'input': payload, 'output': 'No portfolio provided'}
        if not payload.get('org'):
            return {'success': False, 'function': function, 'input': payload, 'output': 'No org provided'}
        if not payload.get('case_group'):
            return {'success': False, 'function': function, 'input': payload, 'output': 'No case group provided'}
        intent = payload.get('intent')
        if not intent or not isinstance(intent, dict):
            return {'success': False, 'function': function, 'input': payload, 'output': 'No intent provided'}

        portfolio = payload['portfolio']
        org = payload['org']
        case_group = payload['case_group']
        init = payload.get('_init') or {}
        plan_actions = init.get('plan_actions') if isinstance(init, dict) else None

        try:
            self.AGU = AgentUtilities(
                self.config, portfolio, org,
                payload.get('_entity_type', 'some_entity_type'),
                payload.get('_entity_id', 'some_entity_id'),
                payload.get('_thread', 'some_thread')
            )
            ctx = RequestContext(portfolio=portfolio, org=org, case_group=case_group, init=init)
            self._set_context(ctx)

            llm = AIResponsesLLM(self.AGU)
            prompts = self._load_prompts(portfolio, org, case_group=case_group)
            action_catalog = self._load_actions(portfolio, org)

            if plan_actions:
                if isinstance(plan_actions, list):
                    plan_actions_set = {a.strip() for a in plan_actions if a.strip()}
                else:
                    plan_actions_set = {a.strip() for a in str(plan_actions).split(',') if a.strip()}
                action_catalog = [a for a in action_catalog if a.key in plan_actions_set]

            use_compose_from_cases = init.get('use_compose_from_cases', False)
            if use_compose_from_cases:
                cases = payload.get('cases') or []
                plan = self.compose_from_cases(intent, llm=llm, action_catalog=action_catalog,
                                               prompts=prompts, cases=cases)
            else:
                plan = self.compose_plan_light(intent, action_catalog=action_catalog)

            return {
                'success': True,
                'function': function,
                'input': payload,
                'output': {
                    'intent': intent,
                    'plan': asdict(plan),
                },
            }
        except Exception as e:
            print(f'ProposePlan failed: {e}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR: {str(e)}'}
