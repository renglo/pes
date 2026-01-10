#
from renglo.data.data_controller import DataController
from renglo.docs.docs_controller import DocsController
from renglo.chat.chat_controller import ChatController
from renglo.schd.schd_controller import SchdController
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config

# Improved Generate and Execute Plan
from pes.handlers.execute_plan import ExecutePlan

# Legacy Generate and Execute Plan
#from .execute_plan import ExecutePlan

import importlib.resources
import yaml


from openai import OpenAI

import random
import json
import boto3
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import re
from decimal import Decimal
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
import uuid
import inspect


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)





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
    


    # -------------------------------------------------------- LOOP FUNCTIONS

    def continuity_router(self, c_id: Optional[str]) -> Dict[str, Any]:
        """
        Route continuity requests based on continuity ID.

        Analyzes the continuity ID (c_id) and determines which action to take:
        generate new plan, initiate plan execution, or resume from a specific point.
        Validates the c_id format and verifies nonce for security.

        Parameters
        ----------
        c_id : str or None
            Continuity ID in format: irn:c_id:<plan_id>:<plan_step>:<action_step>:<tool_step>:<nonce>
            If None or empty, triggers new plan generation

        Returns
        -------
        dict
            {
                'success': bool,
                'function': str,
                'input': str,
                'output': dict
                    {
                        'next_action': str,        # 'generate_plan', 'initiate_plan', 'initiate_action', 'initiate_tool', 'resume_tool'
                        'plan_id': str,            # If available
                        'plan_step': str,          # If available
                        'action_step': str,        # If available
                        'tool_step': str,          # If available
                        'message': str
                    }
            }

        Process
        -------
        1. Checks if c_id exists (if not, returns 'generate_plan')
        2. Validates c_id format (must start with 'irn:c_id')
        3. Parses c_id components (plan_id, plan_step, action_step, tool_step, nonce)
        4. Verifies plan exists in workspace
        5. Validates nonce against action_log for security
        6. Determines next_action based on which components are present

        Notes
        -----
        Resume rules:
        - No c_id or invalid format -> 'generate_plan'
        - Has plan_id but no plan_step -> 'initiate_plan' (start from step 0)
        - Has plan_step but no action_step -> 'initiate_action' (start action 0)
        - Has action_step but no tool_step -> 'initiate_tool' (start tool 0)
        - Has all components -> 'resume_tool' (continue from tool_step)

        Security:
        - Nonce is validated against action_log to prevent replay attacks
        - Format validation prevents malformed c_ids

        """
        try :

            function = inspect.currentframe().f_code.co_name

            # Check if continuity id exists
            if not c_id:
                # Send to the planner
                self.AGU.print_chat(f'No c_id provided...','transient')
                next_action = 'to_be_determined'
                
                return {
                        'success':True,
                        'function':function,
                        'input': c_id,
                        'output':{
                            'next_action':next_action,
                            'message':'No continuity id, Creating new plan'
                        }
                }


            # Check if the continuity id (c_id) has the right format
            parts = c_id.split(':')

            if parts[0] != 'irn' or parts[1] != 'c_id':
                pr = 'The continuity id is not valid.'
                self.AGU.print_chat(pr,'transient')
                next_action = 'c_id_error'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                            'next_action':next_action,
                            'message':pr
                        }    
                }

            # PLAN ID
            if not parts[2]:
                # send to planner
                pr = f'Error: There is no plan_id in the c_id: ({c_id})'
                self.AGU.print_chat(pr,'transient')
                next_action = 'c_id_error'
                
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'message':pr
                    }
                }
                
            
            # There is a plan_id in the c_id 
            plan_id = parts[2]

            # Retrieve plan from workspace
            workspace = self.AGU.get_active_workspace()

            if plan_id not in workspace['plan']:
                
                pr = f'No valid plan, generating new one.'
                self.AGU.print_chat(pr,'transient')
                next_action = 'c_id_error'
                
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'message':pr
                    }
                }

            else:
                plan = workspace['plan'][plan_id]
                plan_state = workspace['state_machine'][plan_id]    
                self._update_context(plan=plan,state_machine=plan_state)
                
                
            # PLAN STEP
            if not parts[3]:
                # No plan step
                pr = f'Starting from the first step... ({c_id})'
                self.AGU.print_chat(pr,'transient')
                next_action = 'initiate_plan'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':0,
                        'message':pr
                    }
                }


            # There is a plan step id. Validate it against retrieved plan
            plan_step = parts[3]
            step_exists = False
            for p_step in plan.get('steps', []):
                print(f'Plan_step:',plan_step,type(plan_step))
                print(f'p_step[step_id]:',p_step.get('step_id'),type(p_step.get('step_id')))
                # plan_step is already a string from split, only convert step_id to string
                step_id = p_step.get('step_id')
                if step_id is not None and plan_step == str(step_id):
                    #Step exists
                    step_exists = True
                    break

            print('Checking step_exist')
            if not step_exists:
                pr = f'There is a plan, the step is not valid, starting from the first step'
                self.AGU.print_chat(pr,'transient')
                next_action = 'initiate_plan'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':0,
                        'message':pr
                    }
                }

            print('Checking action_step')
            if not parts[4]:
                # There is a plan step but no action step. Starting the new action execution
                pr = f'Starting the new action execution... ({c_id})'
                self.AGU.print_chat(pr,'transient')
                next_action = 'initiate_action'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':plan_step,
                        'action_step':0,
                        'message':pr
                    }  
                }


            # There is a plan step and an action step.
            action_step = parts[4]

            print('Checking tool step')
            # If no tool
            if not parts[5]:
                # Tool can start from the beginning
                pr = f'Starting tool... ({c_id})'
                self.AGU.print_chat(pr,'transient')
                next_action = 'initiate_tool'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':plan_step,
                        'action_step':action_step,
                        'tool_step':0,  
                        'message':pr 
                    }
                }


            #Tool should resume where it left
            tool_step = parts[5]
            print(f'Original tool_step: {tool_step}')
            print(f'Nonce: {parts[6]}',type(parts[6]))

            # If tool_step exists, there should be a nonce in the state machine that matches it.
            if not parts[6]:
                print('No nonce, making tool_step = 0')
                # No nonce
                tool_step = 0
            else:
                # Check if nonce matches state machine
                # Find the last entry in action_log that has a nonce attribute
                # Find the step by step_id (plan_step is a step_id, not a list index)
                step_state = None
                for step in plan_state['steps']:
                    if str(step.get('step_id')) == plan_step:
                        step_state = step
                        break

                if step_state is None:
                    print(f'Step with step_id {plan_step} not found in plan_state, resetting tool_step = 0')
                    tool_step = 0
                else:
                    action_log = step_state.get('action_log', [])
                    nonce = None
                    for entry in reversed(action_log):
                        if 'nonce' in entry:
                            nonce = entry['nonce']
                            break

                    if nonce is None:
                        # No nonce found in action_log, reset call
                        print('No nonce found in action_log, resetting tool_step = 0')
                        tool_step = 0
                    else:
                        print(type(nonce))
                        if int(nonce) != int(parts[6]):
                            # InValid nonce, reset call
                            print('Invalid nonce, resetting tool_step = 0')
                            tool_step = 0
                        else:
                            print(f'Nonce has been verified, tool_step = {tool_step}')



            # Resuming the tool execution
            next_action = 'resume_tool'
            pr = f'There is a plan and a step and an action and a tool resume point... ({c_id})'
            self.AGU.print_chat(pr,'transient')

            return {
                'success':True,
                'input': c_id,
                'output':{
                    'next_action':next_action,
                    'plan_id':plan_id,
                    'plan_step':plan_step,
                    'action_step':action_step,
                    'tool_step':tool_step,
                    'message':pr     
                }
            }

        except Exception as e:
            pr = f'🤖❌ @continuity_router:{e}'
            self.AGU.print_chat(pr,'error')

            return {
                        'success':True,
                        'function':function,
                        'input': c_id,
                        'output':{
                            'next_action':'to_be_determined',
                            'message':pr
                        }
                }



    def run_plan_action_tool(
        self,
        plan_id: str,
        plan_step: str,
        action_step: str,
        tool_step: str
    ) -> Dict[str, Any]:
        """
        Execute a specific plan step with action and tool.

        Instantiates ExecutePlan and executes the specified plan step, action,
        and tool combination. This is called after continuity_router determines
        where to resume execution.

        Parameters
        ----------
        plan_id : str
            Plan ID to execute
        plan_step : str
            Plan step ID (step_id)
        action_step : str
            Action step (tool name or "*")
        tool_step : str
            Tool step status (0-7)

        Returns
        -------
        dict
            {
                'success': bool,
                'function': str,
                'input': dict,                 # Payload with plan_id, plan_step, action_step, tool_step
                'output': dict                  # Execution result from ExecutePlan
            }

        Process
        -------
        1. Instantiates ExecutePlan with AgentUtilities
        2. Creates payload with plan_id, plan_step, action_step, tool_step
        3. Calls ExecutePlan.run()
        4. Returns execution result

        Notes
        -----
        This method delegates to ExecutePlan which handles the actual execution
        of plan steps, including calling Specialist for ReAct loops.

        See Also
        --------
        ExecutePlan : Plan execution handler
        continuity_router : Determines which plan/step/action/tool to execute
        """
        function = inspect.currentframe().f_code.co_name
        print('Running:',function)

        try:
            # 1. Instantiate Executor
            executor = ExecutePlan(self.AGU)

            payload = {
                'plan_id':plan_id,
                'plan_step':plan_step,
                'action_step':action_step,
                'tool_step':tool_step
            }

            # 2. Execute plan
            execution_result = executor.run(payload)
            if not execution_result['success']:
                return {
                    'success':False,
                    'function':function,
                    'input':payload,
                    'output':execution_result['output']
                }
                

            print(json.dumps(execution_result, indent=2, cls=DecimalEncoder))

            return {
                'success':True,
                'function': function,
                'input':payload,
                'output':execution_result
            }

        except Exception as e:
            pr = f'🤖❌ @rpat:{e}'
            self.AGU.print_chat(pr,'error')

            return {
                'success':False,
                'function':function,
                'input':payload,
                'output':pr
            }
          
    def execute_plan(self,next_action,plan_id,plan_step,action_step,tool_step):
         
        function = 'execute_plan'
        # We pass the plan_id and not the plan itself. Executor will retrieve it from workspace
        self.AGU.print_chat(f'Calling the executor for plan:{plan_id}, plan_step:{plan_step}, action_step:{action_step}, tool_step:{tool_step}','transient')
        
        if next_action == 'initiate_plan':
            response = self.run_plan_action_tool(plan_id,0,0,0)
        elif next_action == 'initiate_action':
            response = self.run_plan_action_tool(plan_id,plan_step,0,0)
        elif next_action == 'initiate_tool':
            response = self.run_plan_action_tool(plan_id,plan_step,action_step,0)
        elif next_action == 'resume_tool':
            response = self.run_plan_action_tool(plan_id,plan_step,action_step,tool_step) 
            
        execution_results = json.dumps(response, indent=2, cls=DecimalEncoder)
        return {'success':True,'function':function,'input':f'{plan_id}:{plan_step}:{action_step}:{tool_step}','output':response}   

    

        
    ## Execution of Intentions
    def act(self,execution_request, extra=None):
        action = 'act'
        
        list_tools_raw = self._get_context().list_tools
        
        list_handlers = {}
        for t in list_tools_raw:
            list_handlers[t.get('key', '')] = t.get('handler', '')
            
        self._update_context(list_handlers=list_handlers)
    
        """Execute the current intention and return standardized response"""
        try:
            
            tool_name = execution_request['tool_calls'][0]['function']['name']
            params = execution_request['tool_calls'][0]['function']['arguments']
            if isinstance(params, str):
                params = json.loads(params)
            tid = execution_request['tool_calls'][0]['id']
            
            print(f'tid:{tid}')

            if not tool_name:
                raise ValueError("❌ No tool name provided in tool selection")
                
            print(f"Selected tool: {tool_name}")
            self.AGU.print_chat(f'Calling tool {tool_name} with parameters {params} ', 'error')
            print(f"Parameters: {params}")

            # Check if handler exists
            if tool_name not in list_handlers:
                error_msg = f"❌ No handler found for tool '{tool_name}'"
                print(error_msg)
                self.AGU.print_chat(error_msg, 'error')
                raise ValueError(error_msg)
            
            # Check if handler is an empty string
            if list_handlers[tool_name] == '':
                error_msg = f"❌ Handler is empty"
                print(error_msg)
                self.AGU.print_chat(error_msg, 'error')
                raise ValueError(error_msg)
                
            # Check if handler has the right format
            handler_route = list_handlers[tool_name]
            parts = handler_route.split('/')
            if len(parts) != 2:
                error_msg = f"❌ {tool_name} is not a valid tool."
                print(error_msg)
                self.AGU.print_chat(error_msg, 'error')
                raise ValueError(error_msg)
            

            portfolio = self._get_context().portfolio
            org = self._get_context().org
            
            params['_portfolio'] = self._get_context().portfolio
            params['_org'] = self._get_context().org
            params['_entity_type'] = self._get_context().entity_type
            params['_entity_id'] = self._get_context().entity_id
            params['_thread'] = self._get_context().thread
            
            # Add the extra parameters to the params object
            if extra and isinstance(extra, dict):
                params.update(extra) 
            
            print(f'Calling {handler_route} ') 
            
            response = self.SHC.handler_call(portfolio,org,parts[0],parts[1],params)
            
            print(f'Handler response:{response}')

            if not response['success']:
                return {'success':False,'action':action,'input':params,'output':response}

            # The response of every handler always comes nested 
            clean_output = response['output']
            clean_output_str = json.dumps(clean_output, cls=DecimalEncoder)
            
            interface = None
            
            # The handler determines the interface
            if isinstance(response['output'], dict) and 'interface' in response['output']:
                interface = response['output']['interface']
            elif isinstance(response['output'], list) and len(response['output']) > 0 and 'interface' in response['output'][0]:
                interface = response['output'][0]['interface']

               
            
            tool_out = {
                    "role": "tool",
                    "tool_call_id": f'{tid}',
                    "content": clean_output_str,
                    "tool_calls":False
                }
            

            # Save the message after it's created
            if interface:
                self.AGU.save_chat(tool_out,interface=interface,connection_id=self._get_context().connection_id)
                
            else:
                self.AGU.save_chat(tool_out,connection_id=self._get_context().connection_id)
                

            # Results coming from the handler
            self._update_context(execute_intention_results=tool_out)

            # Save handler result to workspace
            
            # Turn an object like this one: {"people":"4","time":"16:00","date":"2025-06-04"}
            # Into a string like this one: "4/16:00/2026-06-04"
            # If the value of each key is not a string just output an empty space in its place
            #params_str = self.format_object_to_slash_string(params)
            index = f'irn:tool_rs:{handler_route}' 
            tool_input = execution_request['tool_calls'][0]['function']['arguments'] 
            #input is a serialize json, you need to turn it into a python object before inserting it into the value dictionary
            tool_input_obj = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            value = {'input': tool_input_obj, 'output': clean_output}
            self.AGU.mutate_workspace({'cache': {index:value}}, public_user=self._get_context().public_user, workspace_id=self._get_context().workspace_id)
            
            print(f'flag5')
            
            #print(f'message output: {tool_out}')
            print("✅ Tool execution complete.")
            
            return {"success": True, "action": action, "input": execution_request, "output": tool_out}
                    
        except Exception as e:

            error_msg = f"❌ Execute Intention failed. @act trying to run tool:'{tool_name}': {str(e)}"
            self.AGU.print_chat(error_msg,'error') 
            print(error_msg)
            self._update_context(execute_intention_error=error_msg)
            
            error_result = {
                "success": False, "action": action,"input": execution_request,"output": str(e)    
            }
            
            self._update_context(execute_intention_results=error_result)
            return error_result
        
    
    
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

            if 'next' in payload:
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
           
            results = []

            
            # Step 0: Create thread/message document
            print('Creating document for this turn ...')
            response_0 = self.AGU.new_chat_message_document(context.message, public_user=context.public_user, next=context.next)
            results.append(response_0)
            if not response_0['success']:
                return {'success':False,'action':action,'output':results}


            # Step 1: Continuity Router
            self.AGU.print_chat(f'Analyzing continuity ID...','transient')
            
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
            if self._get_context().plan and self._get_context().state_machine :             
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
                
                # Step 1: Interpret. We receive the message from the user and we issue a tool command or another message       
                response_2 = self.AGU.interpret(list_actions=context.list_actions,list_tools=context.list_tools)
                results.append(response_2)
                if not response_2['success']:
                    # Something went wrong during message interpretation
                    return {'success':False,'action':action,'output':results}  
                       
                # Check whether we need to run a tool
                if 'tool_calls' not in response_2['output'] or not response_2['output']['tool_calls']:
                    # No tool needs execution. 
                    # Most likely the agent is asking for more information to fill tool parameters. 
                    # Or agent is answering questions directly from the belief system.
                    self.AGU.print_chat(f'🤖','text')
                    return {'success':True,'action':action,'input':payload,'output':results}
                                
                else:
                    # Step 2: Act. Agent runs the tool
                    extra = {'case_group':'x'}
                    
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
            self.AGU.print_chat(f'🤖❌:{e}','error')
            error_result = {'success':False,'action':action,'output':f'Plan Generation Run failed. Error:{str(e)}'}
            results.append(error_result)
            return {'success':False,'action':action,'output':f'Plan Generation Run failed. Error:{str(e)}','stack':results}
        

        


# Test block
if __name__ == '__main__':
    # Creating an instance
    pass
