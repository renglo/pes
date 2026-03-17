# remediation_agent.py
"""
Custom PES for remediation: orchestrates plan generation (case_group='remediation',
logs from message) and plan execution. Standalone implementation following the
agent_trips pattern (no inheritance from PesAgent).
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from contextvars import ContextVar
from decimal import Decimal
import json
import inspect

from renglo.data.data_controller import DataController
from renglo.docs.docs_controller import DocsController
from renglo.chat.chat_controller import ChatController
from renglo.schd.schd_controller import SchdController
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config

from pes.handlers.execute_plan import ExecutePlan


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)


@dataclass
class RequestContext:
    """Request-scoped context for remediation agent (same shape as PES, with case_group)."""
    connection_id: str = ''
    portfolio: str = ''
    org: str = ''
    public_user: str = ''
    entity_type: str = ''
    entity_id: str = ''
    thread: str = ''
    workspace_id: str = ''
    chat_id: str = ''
    workspace: Dict[str, Any] = field(default_factory=dict)
    belief: Dict[str, Any] = field(default_factory=dict)
    desire: str = ''
    action: str = ''
    plan: Dict[str, Any] = field(default_factory=dict)
    execute_intention_results: Dict[str, Any] = field(default_factory=dict)
    execute_intention_error: str = ''
    completion_result: Dict[str, Any] = field(default_factory=dict)
    list_handlers: Dict[str, Any] = field(default_factory=dict)
    list_actions: List[Dict[str, Any]] = field(default_factory=list)
    list_tools: List[Dict[str, Any]] = field(default_factory=list)
    next: Optional[str] = None
    message: str = ''
    case_group: str = ''
    state_machine: Dict[str, Any] = field(default_factory=dict)


request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())


class RemediationAgent:
    """
    Custom PES for remediation: same flow as PesAgent but standalone (no inheritance).
    case_group is always 'remediation'; logs for generate_plan come from context.message (payload['data']).
    """

    def __init__(self):
        self.config = load_config()
        self.DAC = DataController(config=self.config)
        self.DCC = DocsController(config=self.config)
        self.CHC = ChatController(config=self.config)
        self.SHC = SchdController(config=self.config)
        self.AGU = None

    def _get_context(self) -> RequestContext:
        return request_context.get()

    def _set_context(self, context: RequestContext):
        request_context.set(context)

    def _update_context(self, **kwargs):
        context = self._get_context()
        for key, value in kwargs.items():
            setattr(context, key, value)
        self._set_context(context)

    def continuity_router(self, c_id: Optional[str]) -> Dict[str, Any]:
        """Route continuity requests based on continuity ID (same logic as pes_agent)."""
        try:
            function = inspect.currentframe().f_code.co_name

            if not c_id:
                self.AGU.print_chat('No c_id provided...', 'transient')
                return {
                    'success': True,
                    'function': function,
                    'input': c_id,
                    'output': {'next_action': 'to_be_determined', 'message': 'No continuity id, Creating new plan'},
                }

            parts = c_id.split(':')
            if parts[0] != 'irn' or parts[1] != 'c_id':
                self.AGU.print_chat('The continuity id is not valid.', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {'next_action': 'c_id_error', 'message': 'The continuity id is not valid.'},
                }

            if not parts[2]:
                self.AGU.print_chat(f'Error: There is no plan_id in the c_id: ({c_id})', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {'next_action': 'c_id_error', 'message': 'No plan_id in c_id'},
                }

            plan_id = parts[2]
            workspace = self.AGU.get_active_workspace()

            if plan_id not in workspace.get('plan', {}):
                self.AGU.print_chat('No valid plan, generating new one.', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {'next_action': 'c_id_error', 'message': 'No valid plan'},
                }

            plan = workspace['plan'][plan_id]
            plan_state = workspace.get('state_machine', {}).get(plan_id, {})
            self._update_context(plan=plan, state_machine=plan_state)

            if not parts[3]:
                self.AGU.print_chat(f'Starting from the first step... ({c_id})', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {'next_action': 'initiate_plan', 'plan_id': plan_id, 'plan_step': 0, 'message': 'Starting from first step'},
                }

            plan_step = parts[3]
            step_exists = False
            for p_step in plan.get('steps', []):
                step_id = p_step.get('step_id')
                if step_id is not None and plan_step == str(step_id):
                    step_exists = True
                    break

            if not step_exists:
                self.AGU.print_chat('Step not valid, starting from the first step', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {'next_action': 'initiate_plan', 'plan_id': plan_id, 'plan_step': 0, 'message': 'Invalid step'},
                }

            if not parts[4]:
                self.AGU.print_chat(f'Starting the new action execution... ({c_id})', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {
                        'next_action': 'initiate_action',
                        'plan_id': plan_id,
                        'plan_step': plan_step,
                        'action_step': 0,
                        'message': 'Starting action',
                    },
                }

            action_step = parts[4]

            if not parts[5]:
                self.AGU.print_chat(f'Starting tool... ({c_id})', 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {
                        'next_action': 'initiate_tool',
                        'plan_id': plan_id,
                        'plan_step': plan_step,
                        'action_step': action_step,
                        'tool_step': 0,
                        'message': 'Starting tool',
                    },
                }

            tool_step = parts[5]
            if not parts[6]:
                tool_step = 0
            else:
                step_state = None
                for step in plan_state.get('steps', []):
                    if str(step.get('step_id')) == plan_step:
                        step_state = step
                        break
                if step_state is None:
                    tool_step = 0
                else:
                    action_log = step_state.get('action_log', [])
                    nonce = None
                    for entry in reversed(action_log):
                        if 'nonce' in entry:
                            nonce = entry['nonce']
                            break
                    if nonce is None or int(nonce) != int(parts[6]):
                        tool_step = 0

            self.AGU.print_chat(f'Resuming tool execution... ({c_id})', 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'resume_tool',
                    'plan_id': plan_id,
                    'plan_step': plan_step,
                    'action_step': action_step,
                    'tool_step': tool_step,
                    'message': 'Resuming tool',
                },
            }

        except Exception as e:
            pr = f'@continuity_router:{e}'
            self.AGU.print_chat(pr, 'error')
            return {
                'success': True,
                'function': inspect.currentframe().f_code.co_name,
                'input': c_id,
                'output': {'next_action': 'to_be_determined', 'message': pr},
            }

    def run_plan_action_tool(
        self,
        plan_id: str,
        plan_step: Any,
        action_step: Any,
        tool_step: Any,
    ) -> Dict[str, Any]:
        """Execute a specific plan step via ExecutePlan."""
        function = inspect.currentframe().f_code.co_name
        payload = {
            'plan_id': plan_id,
            'plan_step': plan_step,
            'action_step': action_step,
            'tool_step': tool_step,
        }
        try:
            executor = ExecutePlan(self.AGU)
            execution_result = executor.run(payload)
            if not execution_result.get('success'):
                return {
                    'success': False,
                    'function': function,
                    'input': payload,
                    'output': execution_result.get('output'),
                }
            return {
                'success': True,
                'function': function,
                'input': payload,
                'output': execution_result,
            }
        except Exception as e:
            self.AGU.print_chat(f'@rpat:{e}', 'error')
            return {'success': False, 'function': function, 'input': payload, 'output': str(e)}

    def execute_plan(
        self,
        next_action: str,
        plan_id: str,
        plan_step: Any,
        action_step: Any,
        tool_step: Any,
    ) -> Dict[str, Any]:
        """Dispatch to run_plan_action_tool with the right indices."""
        self.AGU.print_chat(
            f'Calling executor plan:{plan_id}, plan_step:{plan_step}, action_step:{action_step}, tool_step:{tool_step}',
            'transient',
        )
        if next_action == 'initiate_plan':
            response = self.run_plan_action_tool(plan_id, 0, 0, 0)
        elif next_action == 'initiate_action':
            response = self.run_plan_action_tool(plan_id, plan_step, 0, 0)
        elif next_action == 'initiate_tool':
            response = self.run_plan_action_tool(plan_id, plan_step, action_step, 0)
        elif next_action == 'resume_tool':
            response = self.run_plan_action_tool(plan_id, plan_step, action_step, tool_step)
        else:
            response = {'success': False, 'function': 'execute_plan', 'input': '', 'output': 'Unknown next_action'}
        return {
            'success': response.get('success', False),
            'function': 'execute_plan',
            'input': f'{plan_id}:{plan_step}:{action_step}:{tool_step}',
            'output': response,
        }

    def act(self, execution_request: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the current intention (tool call) and return standardized response. Same logic as pes_agent."""
        action = 'act'
        list_tools_raw = self._get_context().list_tools
        list_handlers = {}
        list_inits = {}
        for t in list_tools_raw:
            list_handlers[t.get('key', '')] = t.get('handler', '')
            init_value = t.get('init', {})
            if isinstance(init_value, str):
                try:
                    init_value = json.loads(init_value)
                except (json.JSONDecodeError, ValueError):
                    init_value = {}
            list_inits[t.get('key', '')] = init_value if isinstance(init_value, dict) else {}
        self._update_context(list_handlers=list_handlers)

        try:
            tool_name = execution_request['tool_calls'][0]['function']['name']
            params = execution_request['tool_calls'][0]['function']['arguments']
            if isinstance(params, str):
                params = json.loads(params)
            tid = execution_request['tool_calls'][0]['id']

            if not tool_name:
                raise ValueError('No tool name provided in tool selection')

            self.AGU.print_chat(f'Calling tool {tool_name} with parameters {params}', 'transient')

            if tool_name not in list_handlers:
                raise ValueError(f"No handler found for tool '{tool_name}'")
            if list_handlers[tool_name] == '':
                raise ValueError('Handler is empty')

            handler_init = {}
            if isinstance(list_inits.get(tool_name), dict):
                handler_init = list_inits[tool_name]

            handler_route = list_handlers[tool_name]
            parts = handler_route.split('/')
            if len(parts) != 2:
                raise ValueError(f'{tool_name} is not a valid tool.')

            params['_portfolio'] = self._get_context().portfolio
            params['_org'] = self._get_context().org
            params['_entity_type'] = self._get_context().entity_type
            params['_entity_id'] = self._get_context().entity_id
            params['_thread'] = self._get_context().thread
            params['_init'] = handler_init

            if extra and isinstance(extra, dict):
                params.update(extra)

            response = self.SHC.handler_call(self._get_context().portfolio, self._get_context().org, parts[0], parts[1], params)

            if not response.get('success'):
                return {'success': False, 'action': action, 'input': params, 'output': response}

            clean_output = response.get('output')
            clean_output_str = json.dumps(clean_output, cls=DecimalEncoder)

            interface = None
            if isinstance(response.get('output'), dict) and 'interface' in response.get('output', {}):
                interface = response['output']['interface']
            elif isinstance(response.get('output'), list) and len(response.get('output', [])) > 0:
                first = response['output'][0]
                if isinstance(first, dict) and 'interface' in first:
                    interface = first['interface']

            tool_out = {
                'role': 'tool',
                'tool_call_id': tid,
                'content': clean_output_str,
                'tool_calls': False,
            }

            if interface:
                self.AGU.save_chat(tool_out, interface=interface, connection_id=self._get_context().connection_id)
            else:
                self.AGU.save_chat(tool_out, connection_id=self._get_context().connection_id)

            self._update_context(execute_intention_results=tool_out)

            index = f'irn:tool_rs:{handler_route}'
            tool_input = execution_request['tool_calls'][0]['function']['arguments']
            tool_input_obj = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            value = {'input': tool_input_obj, 'output': clean_output}
            self.AGU.mutate_workspace(
                {'cache': {index: value}},
                public_user=self._get_context().public_user,
                workspace_id=self._get_context().workspace_id,
            )

            return {'success': True, 'action': action, 'input': execution_request, 'output': tool_out}

        except Exception as e:
            tool_name = (execution_request.get('tool_calls') or [{}])[0].get('function', {}).get('name', '?')
            error_msg = f'Execute Intention failed. @act tool:{tool_name}: {str(e)}'
            self.AGU.print_chat(error_msg, 'error')
            self._update_context(execute_intention_error=error_msg)
            return {
                'success': False,
                'action': action,
                'input': execution_request,
                'output': str(e),
            }

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for remediation.

        Contract:
        - The caller sends the remediation logs in payload['data'].
        - context.message is set to payload['data'].
        - belief['_remediation_logs'] stores payload['data'] and is what generate_plan uses as logs.
        - case_group is always 'remediation'.

        For generate_plan, use the logs from belief['_remediation_logs'] (i.e., payload['data']).
        For commit_plan, pass plan_cache_key "irn:tool_rs:pes/generate_plan".
        """
        action = 'run > remediation_agent'
        results = []
        payload = dict(payload)

        try:
            context = RequestContext()

            if 'connectionId' in payload:
                context.connection_id = payload['connectionId']
            if 'portfolio' in payload:
                context.portfolio = payload['portfolio']
            else:
                return {'success': False, 'action': action, 'message': 'No portfolio provided', 'input': payload, 'output': results}
            if 'org' in payload:
                context.org = payload['org']
            else:
                context.org = '_all'
            if 'public_user' in payload:
                context.public_user = payload['public_user']
            if 'entity_type' in payload:
                context.entity_type = payload['entity_type']
            else:
                context.entity_type = 'remediation'
            if 'entity_id' in payload:
                context.entity_id = payload['entity_id']
            else:
                context.entity_id = 'remediation_entity_id'
            if 'thread' in payload:
                context.thread = payload['thread']
            else:
                context.thread = 'remediation_thread'
            if 'workspace' in payload:
                context.workspace_id = payload['workspace']
            if 'next' in payload:
                context.next = payload['next']
            else:
                context.next = None
            if 'data' in payload:
                context.message = payload['data']

            context.case_group = 'remediation'
            context.belief = dict(context.belief or {})
            # Logs for generate_plan: remediation_agent expects logs only in payload['data']
            context.belief['_remediation_logs'] = payload.get('data')

            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                context.entity_type,
                context.entity_id,
                context.thread,
                connection_id=context.connection_id,
            )

            actions_resp = self.DAC.get_a_b(context.portfolio, context.org, 'schd_actions')
            context.list_actions = actions_resp.get('items') or []
            tools_resp = self.DAC.get_a_b(context.portfolio, context.org, 'schd_tools')
            context.list_tools = tools_resp.get('items') or []

            self._set_context(context)

            response_0 = self.AGU.new_chat_message_document(
                context.message, public_user=context.public_user, next=context.next
            )
            results.append(response_0)
            if not response_0.get('success'):
                return {'success': False, 'action': action, 'input': payload, 'output': results}

            self.AGU.print_chat('Analyzing continuity ID...', 'transient')
            response_1 = self.continuity_router(context.next)
            results.append(response_1)
            if not response_1.get('success'):
                return {'success': False, 'action': action, 'input': payload, 'output': results}

            next_action = response_1.get('output', {}).get('next_action')
            plan_id = response_1.get('output', {}).get('plan_id')
            plan_step = response_1.get('output', {}).get('plan_step', 0)
            action_step = response_1.get('output', {}).get('action_step', 0)
            tool_step = response_1.get('output', {}).get('tool_step', 0)

            if next_action in ('initiate_plan', 'initiate_action', 'initiate_tool', 'resume_tool'):
                response_1a = self.execute_plan(next_action, plan_id, plan_step, action_step, tool_step)
                results.append(response_1a)
                self.AGU.save_chat({'role': 'assistant', 'content': 'OK'})
                return {'success': True, 'action': action, 'input': payload, 'output': results}

            if self._get_context().plan and self._get_context().state_machine:
                current_action = 'modifying_plan'
                current_desire = 'Modify an existing plan'
            else:
                current_action = 'creating_plan'
                current_desire = 'Create a new plan'

            self.AGU.mutate_workspace({'action': current_action, 'desire': current_desire})

            loop_limit = 6
            for loops in range(1, loop_limit + 1):
                ctx = self._get_context()
                list_actions_specific = [a for a in ctx.list_actions if a.get('key') == current_action]
                list_tools_specific = []
                if list_actions_specific:
                    tools_ref = list_actions_specific[0].get('tools_reference') or ''
                    if tools_ref and tools_ref not in ('_', '-', '.', ''):
                        tool_keys = [k.strip() for k in tools_ref.split(',') if k.strip()]
                        list_tools_specific = [t for t in ctx.list_tools if t.get('key') in tool_keys]

                response_2 = self.AGU.interpret(
                    list_actions=list_actions_specific,
                    list_tools=list_tools_specific,
                )
                results.append(response_2)
                if not response_2.get('success'):
                    return {'success': False, 'action': action, 'input': payload, 'output': results}

                out = response_2.get('output') or {}
                tool_calls = out.get('tool_calls')
                if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
                    self.AGU.print_chat('OK', 'transient')
                    return {'success': True, 'action': action, 'input': payload, 'output': results}

                first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                func = first.get('function') or {}
                tool_name = func.get('name') or ''
                if not tool_name:
                    return {
                        'success': False,
                        'action': action,
                        'error': 'Invalid tool_calls: function.name missing',
                        'input': payload,
                        'output': results,
                    }

                extra = {'case_group': 'remediation'}
                if tool_name == 'generate_plan':
                    logs = ctx.belief.get('_remediation_logs')
                    if logs is not None:
                        extra['logs'] = logs
                        extra['enriched_log'] = logs
                elif tool_name == 'commit_plan':
                    extra['plan_cache_key'] = 'pes/generate_plan'

                response_3 = self.act(out, extra=extra)
                results.append(response_3)
                if not response_3.get('success'):
                    return {'success': False, 'action': action, 'input': payload, 'output': results}

                if tool_name == 'commit_plan' and 'content' in (response_3.get('output') or {}):
                    try:
                        content = response_3['output']['content']
                        tool_response = json.loads(content) if isinstance(content, str) else content
                        if (
                            isinstance(tool_response, dict)
                            and tool_response.get('next_action') == 'initiate_plan'
                            and tool_response.get('plan_id')
                        ):
                            self.AGU.print_chat('Executing new plan immediately after approval', 'transient')
                            response_3a = self.execute_plan(
                                tool_response['next_action'],
                                tool_response['plan_id'],
                                0, 0, 0,
                            )
                            results.append(response_3a)
                            self.AGU.save_chat({'role': 'assistant', 'content': 'OK'})
                            return {'success': True, 'action': action, 'input': payload, 'output': results}
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass

            response_2 = self.AGU.interpret(no_tools=True)
            results.append(response_2)
            if not response_2.get('success'):
                return {'success': False, 'action': action, 'input': payload, 'output': results}
            self.AGU.print_chat('Can you re-formulate your request please?', 'text')
            return {'success': True, 'action': action, 'input': payload, 'output': results}

        except Exception as e:
            agu = getattr(self, 'AGU', None)
            if agu:
                agu.print_chat(f'(remediation_agent):{e}', 'error')
            return {
                'success': False,
                'action': action,
                'message': f'Remediation PES failed: {e}',
                'input': payload,
                'output': results,
            }


if __name__ == '__main__':
    pass
