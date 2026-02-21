"""
ProposePlan: Handler that takes intent as input and returns a plan.
Intent in, plan out. Called by GeneratePlan and ModifyPlan.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json
from contextvars import ContextVar

from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController


@dataclass
class RequestContext:
    portfolio: str = ''
    org: str = ''
    case_group: str = ''
    init: Dict[str, Any] = field(default_factory=dict)


request_context: ContextVar = ContextVar('propose_plan_context', default=None)


class ProposePlan:
    """
    Handler: intent in, plan out.
    Payload: portfolio, org, case_group, intent, _init (optional)
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
        prompts = {'compose_plan_light': ''}
        try:
            if not case_group:
                raise Exception('No case group')
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
                    if key == 'compose_plan_light' and prompt_text:
                        prompts['compose_plan_light'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts: {e}')
        return prompts

    def _load_actions(self, portfolio: str, org: str, action_ring: str = "schd_actions") -> List:
        from pes.handlers.generate_plan import ActionSpec
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

            from pes.handlers.generate_plan import (
                Planner, ActionSpec,
                SimpleEmbedder, VectorDB, AIResponsesLLM,
            )

            embedder = SimpleEmbedder()
            vdb = VectorDB(embedder)
            llm = AIResponsesLLM(self.AGU)
            prompts = self._load_prompts(portfolio, org, case_group=case_group)
            action_catalog = self._load_actions(portfolio, org)

            if plan_actions:
                if isinstance(plan_actions, list):
                    plan_actions_set = {a.strip() for a in plan_actions if a.strip()}
                else:
                    plan_actions_set = {a.strip() for a in str(plan_actions).split(',') if a.strip()}
                action_catalog = [a for a in action_catalog if a.key in plan_actions_set]

            planner = Planner(vdb=vdb, llm=llm, action_catalog=action_catalog, prompts=prompts)
            plan = planner.compose_plan_light(intent)

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
