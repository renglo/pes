#
from renglo.data.data_controller import DataController
from renglo.docs.docs_controller import DocsController
from renglo.chat.chat_controller import ChatController
from renglo.schd.schd_controller import SchdController
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config

from pes.handlers import utilities as plan_utilities

import importlib.resources
import yaml


from openai import OpenAI

import random
import json
import boto3
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
import uuid



@dataclass
class RequestContext:
    """
    Request-scoped context for agent operations.

    This dataclass stores all context information for a single agent execution,
    including connection details, entity information, workspace state, plan state,
    and execution results. It implements the BDI (Belief-Desire-Intention) model.

    Attributes
    ----------
    connection_id : str
        WebSocket connection ID for responding to user
    portfolio : str
        Portfolio ID
    org : str
        Organization ID
    public_user : str
        External user ID (for messages from outside the system)
    entity_type : str
        Entity type 
    entity_id : str
        Entity ID
    thread : str
        Thread ID
    workspace_id : str
        Workspace ID
    chat_id : str
        Chat ID
    workspace : Dict[str, Any]
        Workspace document with cache and state
    belief : Dict[str, Any]
        Belief state (current knowledge)
    desire : str
        Desire (goal to achieve)
    action : str
        Current action being executed
    plan : Dict[str, Any]
        Current plan document
    execute_intention_results : Dict[str, Any]
        Results from plan execution
    execute_intention_error : str
        Error message from execution (if any)
    completion_result : Dict[str, Any]
        Final completion result
    list_handlers : Dict[str, Any]
        Available handlers registry
    list_actions : List[Dict[str, Any]]
        Available actions list
    list_tools : List[Dict[str, Any]]
        Available tools list
    next : str
        Continuity ID for resuming execution
    message : str
        User message text
    """
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
    next: str = ''
    message: str = ''
    state_machine: Dict[str, Any] = field(default_factory=dict)

# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

class PesAgent:
    """
    Main orchestrator.

    The PES Agent implements a planning and execution system based on plans,
    coordinating between multiple components:
    - GeneratePlan: Generates plans using LLM and case-based reasoning
    - ExecutePlan: Executes plan steps sequentially
    - Specialist: ReAct loop that executes each action
    - Verify: Verifies if actions were completed correctly

    The system uses Continuity IDs (c_id) to allow execution resumption.

    Attributes
    ----------
    config : dict
        System configuration
    DAC : DataController
        Controller for data operations
    DCC : DocsController
        Controller for document operations
    CHC : ChatController
        Controller for chat operations
    SHC : SchdController
        Controller for scheduling operations
    AGU : AgentUtilities, optional
        Agent utilities (initialized in run())


    Notes
    -----
    The PES Agent manages three levels of loops:
    1. Plan Loop: Executes all steps of the plan
    2. Action Loop: ReAct loop within each step (Specialist)
    3. Tool Loop: Loop within each handler for mechanical retry

    Continuity ID format: irn:c_id:<plan_id>:<plan_step>:<action_step>:<tool_step>:<nonce>

    See Also
    --------
    GeneratePlan : Plan generation handler
    ExecutePlan : Plan execution handler
    Specialist : ReAct loop handler
    """

    def __init__(self):
        """
        Initialize PESAgent handler.
        """
        self.config = load_config()
        self.DAC = DataController(config=self.config)
        self.DCC = DocsController(config=self.config)
        self.CHC = ChatController(config=self.config)
        self.SHC = SchdController(config=self.config)

        # AgentUtilities will be initialized in the run function
        self.AGU = None


    def _get_context(self) -> RequestContext:
        """
        Get the current request context.

        Returns
        -------
        RequestContext
            Current request context instance from context variable
        """
        return request_context.get()

    def _set_context(self, context: RequestContext):
        """
        Set the current request context.

        Parameters
        ----------
        context : RequestContext
            Request context instance to set
        """
        request_context.set(context)

    def _update_context(self, **kwargs):
        """
        Update specific fields in the current request context.

        Parameters
        ----------
        **kwargs
            Keyword arguments matching RequestContext field names
        """
        context = self._get_context()
        for key, value in kwargs.items():
            setattr(context, key, value)
        self._set_context(context)

    def _get_utilities(self) -> AgentUtilities:
        """
        Get or create AgentUtilities instance with current context.

        Creates a new AgentUtilities instance if one doesn't exist or if
        the context has changed (different portfolio, org, entity, or thread).

        Returns
        -------
        AgentUtilities
            AgentUtilities instance configured with current context
        """
        context = self._get_context()
        if not self.AGU or (self.AGU.portfolio != context.portfolio or
                           self.AGU.org != context.org or
                           self.AGU.entity_type != context.entity_type or
                           self.AGU.entity_id != context.entity_id or
                           self.AGU.thread != context.thread):
            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                context.entity_type,
                context.entity_id,
                context.thread
            )
        return self.AGU

    def infer_continuity_id(self, user_message: Optional[str] = None) -> Optional[str]:
        """Delegates to :mod:`pes.handlers.utilities`."""
        return plan_utilities.infer_continuity_id(user_message, agu=self.AGU)

    def continuity_router(self, c_id: Optional[str]) -> Dict[str, Any]:
        """Delegates to :mod:`pes.handlers.utilities`."""
        return plan_utilities.continuity_router(
            c_id,
            agu=self.AGU,
            get_workspace=self.AGU.get_active_workspace,
            update_plan_context=lambda p, ps: self._update_context(plan=p, state_machine=ps),
        )

    def run_plan_action_tool(
        self,
        plan_id: str,
        plan_step: str,
        action_step: str,
        tool_step: str,
    ) -> Dict[str, Any]:
        """Delegates to :mod:`pes.handlers.utilities`."""
        return plan_utilities.run_plan_action_tool(
            plan_id, plan_step, action_step, tool_step, agu=self.AGU
        )

    def execute_plan(self, next_action, plan_id, plan_step, action_step, tool_step):
        """Delegates to :mod:`pes.handlers.utilities`."""
        return plan_utilities.execute_plan(
            next_action,
            plan_id,
            plan_step,
            action_step,
            tool_step,
            agu=self.AGU,
        )

    def act(self, execution_request, extra=None):
        """Delegates to :mod:`pes.handlers.utilities`."""
        ctx = self._get_context()
        return plan_utilities.act(
            execution_request,
            agu=self.AGU,
            schc=self.SHC,
            list_tools=ctx.list_tools,
            portfolio=ctx.portfolio,
            org=ctx.org,
            entity_type=ctx.entity_type,
            entity_id=ctx.entity_id,
            thread=ctx.thread,
            context_update=self._update_context,
            public_user=ctx.public_user,
            workspace_id=ctx.workspace_id,
            connection_id=ctx.connection_id,
            extra=extra,
        )


    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for agent execution.

        Orchestrates the complete agent flow: initializes context, creates chat
        message, routes continuity requests, generates plans if needed, and executes
        plans. This is the top-level handler that coordinates all agent components.

        Parameters
        ----------
        payload : dict
            Payload containing:
            {
                'portfolio': str,              # Portfolio ID (required)
                'org': str,                    # Organization ID (optional, defaults to '_all')
                'public_user': str, optional   # External user ID (for external messages)
                'entity_type': str,            # Entity type
                'entity_id': str,              # Entity ID
                'thread': str,                 # Thread ID
                'data': str,                   # User message text
                'connectionId': str, optional  # WebSocket connection ID
                'workspace': str, optional     # Workspace ID
                'next': str, optional          # Continuity ID (c_id) for resuming execution
                'planner_intent': str, optional  # e.g. 'modify' / 'modify_plan' when omitting next
                'planner_mode': str, optional    # alias for planner_intent
            }

        Returns
        -------
        dict
            {
                'success': bool,
                'action': str,                 # 'run'
                'input': dict,                 # Original payload
                'output': list                 # Stack of execution results
            }

        Process
        -------
        1. Initializes RequestContext with payload data
        2. Initializes AgentUtilities (AGU) with context
        3. Creates new chat message document (new_chat_message_document)
        4. Analyzes continuity ID via continuity_router()
        5. Based on continuity router decision:
           - If 'generate_plan': Calls GeneratePlan to create new plan
           - If 'initiate_plan'/'initiate_action'/'initiate_tool'/'resume_tool':
             Calls run_plan_action_tool() to execute plan
        6. Returns results with execution stack

        Notes
        -----
        The agent manages three levels of loops:
        1. Plan Loop: Executes all steps of the plan sequentially
        2. Action Loop: ReAct loop within each step (Specialist)
        3. Tool Loop: Loop within each handler for mechanical retry

        Continuity ID format: irn:c_id:<plan_id>:<plan_step>:<action_step>:<tool_step>:<nonce>


        See Also
        --------
        continuity_router : Routes continuity requests
        run_plan_action_tool : Executes plan steps
        GeneratePlan : Generates new plans
        ExecutePlan : Executes plans
        """
        
        '''
        This agent expects an input with the following format: 
        
        {
            'portfolio':'Portfolio the message is targeted to',
            'org':'Organization the message is targeted to',
            'public_user':'(Optional) External id to identify user in case the message was initiated from outside the system (e.g incoming email or chat messages)',
            'entity_type':'String that shows how to assemble the entity_id. It is also used to group similar chats (it defines the type of chat)',
            'entity_id':'This is the unique id of the chat. It is usually composed from other ids to make it unique. ',
            'thread':'This is the thread ID, needed to locate the workspace and the messages in the thread. A chat can have multiple threads',
            'connectionId': (Optional)'This ID is used to respond back to the user via Websocket. Empty if message came some other way'
            'workspace':'(Optional) Id of the workspace used for this message. A thread can have multiple workspaces.'
            'next':(Optional) ContinuityID sent by the client system to continue a transaction.
        }

        

        Plan of action

        1. Create pre_planner function that checks if the payload comes with a continuation id. If that is the case we'll skip the planner and resume plan execution
        2. Insert Planner. Replace pre_processing function for Planner function
        3. Expose the Actions to the planner for it to assemble the plan (Actions as high level tools)
        4. Store Plan in DB
        5. If plan already exists run de executor.
        6. Pull plan document from DB
        7. Resume execution wherever it left with the help of the continuity id. If no c_id, start the plan from the beginning

            RESUME RULES
            If the c_id shows
            [ p/-/-/- ] Resume from plan_step 0
            [ p/n/-/- ] Resume from plan_step n
            [ p/n/m/- ] Resume from plan_step n , action_step m
            [ p/m/m/q ] Resume from plan_step n , action_step m , tool_step q

        8. Loops
            Plan Loop: This is the loop that runs as long as there are plan_steps
                COS > The goal
                ENGINE > Sequential, one after the other guarded by exist and entry conditions.
            Action Loop: This is the loop that runs as long as there are action_steps (ReAct agent loop)
                COS > The goal of the action_step has been achieved
                ENGINE > A ReAct Agent follows the action steps (optimal path) in a Interpret>Run>Interpret loop
            Tool Loop: This is the loop that runs inside the handler.
                COS > The Tool goal is achieved
                ENGINE > Logic based loop to retry mechanically if error arises.
        '''

        # Initialize a new request context
        action = 'run'
        print(f'Running: {action}')
        print(f'Payload: {payload}')  

        try:

            context = RequestContext()

            # Update context with payload data
            if 'connectionId' in payload:
                context.connection_id = payload["connectionId"]

            if 'portfolio' in payload:
                context.portfolio = payload['portfolio']
            else:
                return {'success':False,'action':action,'input':payload,'output':'No portfolio provided'}

            if 'org' in payload:
                context.org = payload['org']
            else:
                context.org = '_all' #If no org is provided, we switch the Agent to portfolio level

            if 'public_user' in payload:
                context.public_user = payload['public_user']

            if 'entity_type' in payload:
                context.entity_type = payload['entity_type']
            else:
                context.entity_type = 'ag1'
                #return {'success':False,'action':action,'input':payload,'output':'No entity_type provided'}

            if 'entity_id' in payload:
                context.entity_id = payload['entity_id']
            else:
                context.entity_id = '5678a'
                #return {'success':False,'action':action,'input':payload,'output':'No entity_id provided'}

            if 'thread' in payload:
                context.thread = payload['thread']
            else:
                context.thread = '1234c'
                #return {'success':False,'action':action,'input':payload,'output':'No thread provided'}

            if 'workspace' in payload:  # Usually, the workspace id does not come in the payload.
                context.workspace_id = payload['workspace']

            if 'next' in payload and payload['next']:
                context.next = payload['next']
            else:
                context.next = None

            if 'data' in payload:
                context.message = payload['data']


            print('Initializing Agent utilities ...')
            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                context.entity_type,
                context.entity_id,
                context.thread,
                connection_id = context.connection_id
                )
            
            
            # Get available actions and tools
            actions = self.DAC.get_a_b(context.portfolio, context.org, 'schd_actions')
            context.list_actions = actions['items']
            
            tools = self.DAC.get_a_b(context.portfolio, context.org, 'schd_tools')
            context.list_tools = tools['items']
                
 
            # Set the initial context for this turn
            print('Setting initial context ...')
            self._set_context(context)

            skip_continuity_router = False
            if not context.next:
                resolution = plan_utilities.resolve_no_next_continuity(
                    context.message,
                    agu=self.AGU,
                    payload=payload,
                )
                if resolution.mode == 'modify_plan' and resolution.plan and resolution.plan_state:
                    skip_continuity_router = True
                    self._update_context(
                        plan=resolution.plan,
                        state_machine=resolution.plan_state,
                    )
                    self.AGU.print_chat(
                        'Plan modification intent: skipping execution resume router '
                        f'(plan_id={resolution.plan_id}).',
                        'transient',
                    )
                elif resolution.mode == 'resume' and resolution.c_id:
                    context.next = resolution.c_id
                    self._set_context(context)
                    self.AGU.print_chat(
                        f'Inferred continuity id from workspace: {resolution.c_id}.',
                        'transient',
                    )

            results = []

            
            # Step 0: Create thread/message document
            print('Creating document for this turn ...')
            response_0 = self.AGU.new_chat_message_document(context.message, public_user=context.public_user, next=context.next)
            results.append(response_0)
            if not response_0['success']:
                return {'success':False,'action':action,'output':results}


            # Step 1: Continuity Router
            self.AGU.print_chat(f'Analyzing continuity ID...','transient')

            if skip_continuity_router:
                response_1 = {
                    'success': True,
                    'input': None,
                    'output': {
                        'next_action': 'to_be_determined',
                        'message': 'Plan modification path; continuity router skipped.',
                    },
                }
                results.append(response_1)
            else:
                self.AGU.print_chat(f'Running continuity router ...','transient')

                response_1 = self.continuity_router(context.next)
                results.append(response_1)
                if not response_1['success']:
                    return {'success':False,'action':action,'output':results}

                self.AGU.print_chat(f'{response_1['output']['message']} ','transient')
            

            next_action = response_1.get('output', {}).get('next_action', None)
            plan_id = response_1.get('output', {}).get('plan_id', None)
            plan_step = response_1.get('output', {}).get('plan_step', 0)
            action_step = response_1.get('output', {}).get('action_step', 0)
            tool_step = response_1.get('output', {}).get('tool_step', 0)
            
            
            # PLAN EXECUTION  
            if next_action in ['initiate_plan','initiate_action','initiate_tool','resume_tool']:  
                # Skipping high level agent and going directly to plan execution
                response_1a = self.execute_plan(next_action,plan_id,plan_step,action_step,tool_step)
                results.append(response_1a)  
                m = { "role": "assistant", "content":f'🤖🤖'}
                self.AGU.save_chat(m) 
                return {'success':True,'action':action,'input':payload,'output':results}
                
           
            
            # ACTION SELECTION
            # Figure out if a plan already exists to select action.
            if skip_continuity_router:
                current_action = 'modifying_plan'
                current_desire = 'Modify an existing plan'
            elif self._get_context().plan and self._get_context().state_machine:
                # If yes, use the modifying_plan action,
                current_action = 'modifying_plan'
                current_desire = 'Modify an existing plan'
            else:
                # If no, use the creating_plan action.
                current_action = 'creating_plan'
                current_desire = 'Create a new plan'

            print(f'Current Action: {current_action}, Current Desire:{current_desire}')
            self.AGU.mutate_workspace({
                'action': current_action,
                'desire': current_desire
                })
            
            
            # AGENT LOOP
            loops = 0
            loop_limit = 6
            while loops < loop_limit:
                loops = loops + 1
                print(f'Loop iteration {loops}/{loop_limit}')
                
                # Filter actions to only include the current action (security: prevent prompt hacking)
                list_actions_specific = [action for action in context.list_actions if action.get('key') == current_action]
                
                # Get tools_reference from the current action
                list_tools_specific = []
                if list_actions_specific:
                    current_action_obj = list_actions_specific[0]
                    tools_reference = current_action_obj.get('tools_reference', '')
                    
                    # Parse tools_reference (comma-separated string of tool keys)
                    if tools_reference and tools_reference not in ['_', '-', '.', '']:
                        # Split by comma and strip whitespace
                        tool_keys = [key.strip() for key in tools_reference.split(',') if key.strip()]
                        
                        # Filter tools to only include those referenced by the current action
                        list_tools_specific = [
                            tool for tool in context.list_tools 
                            if tool.get('key') in tool_keys
                        ]
                
                # Step 1: Interpret. We receive the message from the user and we issue a tool command or another message       
                response_2 = self.AGU.interpret(list_actions=list_actions_specific,list_tools=list_tools_specific)
                results.append(response_2)
                if not response_2['success']:
                    # Something went wrong during message interpretation
                    return {'success':False,'action':action,'output':results}  
                       
                # Check whether we need to run a tool
                if 'tool_calls' not in response_2['output'] or not response_2['output']['tool_calls']:
                    # No tool needs execution. 
                    # Most likely the agent is asking for more information to fill tool parameters. 
                    # Or agent is answering questions directly from the belief system.
                    self.AGU.print_chat(f'🤖','transient')
                    return {'success':True,'action':action,'input':payload,'output':results}
                                
                else:
                    # Step 2: Act. Agent runs the tool
                    act_ctx = self._get_context()
                    extra: Dict[str, Any] = {'case_group': 'x'}
                    if act_ctx.plan:
                        extra['plan'] = act_ctx.plan
                    if act_ctx.state_machine:
                        extra['state_machine'] = act_ctx.state_machine
                    ws_live = act_ctx.workspace or {}
                    if isinstance(ws_live, dict) and ws_live.get('intent'):
                        extra['intent'] = ws_live['intent']
                    
                    #Validate that response_2['output'] has this format inside : ['tool_calls'][0]['function']['name'] before calling act
                    try:
                        tool_calls = response_2['output'].get('tool_calls', [])
                        if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
                            return {
                                'success': False,
                                'action': action,
                                'error': 'Invalid tool_calls format: tool_calls must be a non-empty list',
                                'output': results
                            }
                        
                        first_tool_call = tool_calls[0]
                        if not isinstance(first_tool_call, dict) or 'function' not in first_tool_call:
                            return {
                                'success': False,
                                'action': action,
                                'error': 'Invalid tool_calls format: first tool_call must have a function key',
                                'output': results
                            }
                        
                        function = first_tool_call['function']
                        if not isinstance(function, dict) or 'name' not in function:
                            return {
                                'success': False,
                                'action': action,
                                'error': 'Invalid tool_calls format: function must have a name key',
                                'output': results
                            }
                        tool_name = function['name']
                    except (KeyError, TypeError, IndexError) as e:
                        return {
                            'success': False,
                            'action': action,
                            'error': f'Invalid response_2 format: {str(e)}',
                            'output': results
                        }
                        
                    
                    
                    response_3 = self.act(response_2['output'], extra = extra)
                    results.append(response_3)
                        
                    if not response_3['success']:
                        # Something went wrong during tool execution
                        return {'success':False,'action':action,'output':results}
                    
                    else:
                        if tool_name == 'commit_plan':
                            # response_3['output'] is tool_out which has 'content' as a JSON string
                            if 'content' in response_3['output']:
                                try:
                                    content = response_3['output']['content']
                                    # Parse the JSON string
                                    tool_response = json.loads(content) if isinstance(content, str) else content
                                    
                                    # Only proceed if it's a dict with next_action and plan_id
                                    if isinstance(tool_response, dict) and 'next_action' in tool_response and 'plan_id' in tool_response and tool_response['next_action']=='initiate_plan':
                                        # PLAN EXECUTION (immediately after approval)
                                        self.AGU.print_chat('Executing new plan immediately after approval','transient')
                                        response_3a = self.execute_plan(tool_response['next_action'],tool_response['plan_id'],0,0,0)
                                        results.append(response_3a)
                                        m = { "role": "assistant", "content":f'🤖🤖'}
                                        self.AGU.save_chat(m)
                                        return {'success':True,'action':action,'input':payload,'output':results}
                                except (json.JSONDecodeError, KeyError, TypeError):
                                    # Skip if content doesn't match expected format
                                    pass
                        
         
            #Gracious exit. Analyze the last tool run (act()) but you can't issue a new tool_call.
            response_2 = self.AGU.interpret(no_tools=True)
            results.append(response_2)
            if not response_2['success']:
                    # Something went wrong during message interpretation
                    return {'success':False,'action':action,'output':results} 
            
            
            # If we reach here, we hit the loop limit
            print(f'Warning: Reached maximum loop limit ({loop_limit})')
            self.print_chat(f'🤖⚠️  Can you re-formulate your request please?','text')
            return {'success':True,'action':action,'input':payload,'output':results}
                
        except Exception as e:
            self.AGU.print_chat(f'🤖❌(pes_agent):{e}','error')
            error_result = {'success':False,'action':action,'output':f'Plan Generation Run failed. Error:{str(e)}'}
            results.append(error_result)
            return {'success':False,'action':action,'output':f'Plan Generation Run failed. Error:{str(e)}','stack':results}
        

        


# Test block
if __name__ == '__main__':
    # Creating an instance
    pass
