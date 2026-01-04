#
from renglo.data.data_controller import DataController
from renglo.docs.docs_controller import DocsController
from renglo.chat.chat_controller import ChatController
from renglo.schd.schd_controller import SchdController
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config

# Improved Generate and Execute Plan
from pes.handlers.execute_plan import ExecutePlan
from pes.handlers.generate_plan import GeneratePlan

# Legacy Generate and Execute Plan
#from .execute_plan import ExecutePlan
#from .generate_plan import GeneratePlan

import importlib.resources
import yaml


from openai import OpenAI

import random
import json
import boto3
from datetime import datetime
from typing import List, Dict, Any, Callable
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
    """Request-scoped context for agent operations."""
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

# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

class HierarchicalAgent:
    def __init__(self,prompts):
        
        self.config = load_config()
        self.DAC = DataController(config=self.config)
        self.DCC = DocsController(config=self.config)
        self.CHC = ChatController(config=self.config)
        self.SHC = SchdController(config=self.config)
        self.prompts = prompts
        
        # AgentUtilities will be initialized in the run function
        self.AGU = None
    

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
        
    def _get_utilities(self) -> AgentUtilities:
        """Get or create AgentUtilities instance with current context."""
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
    
    # NOT USED, SAMPLE FUNCTION TO IMPLEMENT IN CALLING MODULE TO RETRIEVE PROMPTS FROM YAML FILE
    def _load_prompts_from_files(self) -> Dict[str, str]:
        """
        Load prompts from prompts.yaml file in the <calling_module>.prompts package directory.
        Returns a dictionary with keys: 'to_signature', 'adapt_plan', 'compose_plan', 'select_best_plan'
        Uses importlib.resources for proper package resource loading.
        
        This function needs to be implemented in the calling module
        Replace <calling_module> with the real name of the module 
        """
        
        prompts = {
            'to_signature': '',
            'adapt_plan': '',
            'compose_plan': '',
            'select_best_plan': ''
        }
        
        try:
            # Use importlib.resources to access package data files
            # This works whether the package is installed or run from source
            prompts_package = importlib.resources.files('<calling_module>.prompts')
            prompts_yaml_file = prompts_package / 'pes_prompts.yaml'
            
            # Read and parse YAML file
            yaml_content = prompts_yaml_file.read_text(encoding='utf-8')
            data = yaml.safe_load(yaml_content)
            
            # Extract prompts from YAML structure
            if data and 'prompts' in data:
                loaded_prompts = data['prompts']
                for key in prompts.keys():
                    if key in loaded_prompts:
                        prompts[key] = loaded_prompts[key]
                    else:
                        print(f'Warning: Prompt "{key}" not found in prompts.yaml')
            else:
                print('Warning: Invalid prompts.yaml structure - missing "prompts" key')
        except FileNotFoundError:
            print('Warning: prompts.yaml file not found in <calling_module>.prompts package')
        except yaml.YAMLError as e:
            print(f'Warning: Error parsing prompts.yaml: {str(e)}')
        except Exception as e:
            print(f'Warning: Could not load prompts.yaml: {str(e)}')
        
        return prompts
        



    # -------------------------------------------------------- LOOP FUNCTIONS
    
    def continuity_router(self,c_id):
        
        
        try :
            
            function = inspect.currentframe().f_code.co_name
            '''
            This function analyzes the continuity id and outputs an object that shows what function needs to be called.
            
            c_id FORMAT
            irn:c_id:<plan>:<plan_step>:<action_step>:<tool_step>:<nonce>
            
            RESUME RULES
                If the c_id shows 
                [ -:-:-:- ] Send to planner
                [ p:-:-:- ] Resume from plan_step 0 
                [ p:n:-:- ] Resume from plan_step n
                [ p:n:m:- ] Resume from plan_step n , action_step m
                [ p:n:m:q ] Resume from plan_step n , action_step m , tool_step q 
                
                [ p:n:m:q:nonce] The fifth position of the c_id is a nonce, a random 6 digit number that is generated along the c_id and is needed to verify that c_id is legit. 
                                    And to avoid replay and lateral attacks among others.
                
            '''
            
            # Check if continuity id exists
            if not c_id: 
                # Send to the planner
                print('CR: No c_id, generate new plan')
                self.AGU.print_chat(f'Generating new plan... ({c_id})','text', connection_id=self._get_context().connection_id)
                next_action = 'generate_plan'
                
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
                next_action = 'create_plan'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                            'next_action':next_action,
                            'message':'The continuity id is corrupt, it does not start with the prefix irn:c_id. Creating new plan'
                        }    
                }
                
            # PLAN ID
            if not parts[2]:
                # send to planner
                pr = f'No plan, generating new one ({c_id})'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
                next_action = 'create_plan'
                
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'message':'No plan id found. Creating new plan'
                    }
                }
                
            
            # There is a plan_id   
            plan_id = parts[2]
            
            # Retrieve plan from workspace
            workspace = self.AGU.get_active_workspace()
            
            if plan_id not in workspace['plan']:
                
                pr = f'No valid plan, generating new one ({c_id})'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
                next_action = 'create_plan'
                
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'message':'No valid plan id found. Creating new plan'
                    }
                }
                
            else:
                plan = workspace['plan'][plan_id]
                plan_state = workspace['state_machine'][plan_id]
                
                
            
            # PLAN STEP
            if not parts[3]:
                # No plan step
                pr = f'Starting from the first step... ({c_id})'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
                next_action = 'initiate_plan'    
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':0,
                        'message':'There is a plan id but no step id, starting from the first step'
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
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
                next_action = 'initiate_plan'
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':0,
                        'message':'There is a plan, the step is not valid, starting from the first step'
                    }
                }
                        
            print('Checking action_step')      
            if not parts[4]:
                # There is a plan step but no action step. Starting the new action execution
                pr = f'Starting the new action execution... ({c_id})'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
                next_action = 'initiate_action' 
                return {
                    'success':True,
                    'input': c_id,
                    'output':{
                        'next_action':next_action,
                        'plan_id':plan_id,
                        'plan_step':plan_step,
                        'action_step':0,
                        'message':'There is a plan step but no action step. Starting the new action execution'
                    }  
                }
                

            # There is a plan step and an action step.
            action_step = parts[4]
            
            print('Checking tool step')
            if not parts[5]:
                #Tool can start from the beginning
                pr = f'Starting tool... ({c_id})'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
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
                        'message':'There is a plan an action but no tool step. Starting the tool execution' 
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
                    
        
            
        
            next_action = 'resume_tool'  
            pr = f'There is a plan and a step and an action and a tool resume point... ({c_id})'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
            
            return {
                'success':True,
                'input': c_id,
                'output':{
                    'next_action':next_action,
                    'plan_id':plan_id,
                    'plan_step':plan_step,
                    'action_step':action_step,
                    'tool_step':tool_step,
                    'message':'There is a plan an action and a tool step. Resuming the tool execution'      
                }
            }
            
        except Exception as e:
            pr = f'🤖❌ @continuity_router:{e}'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
            
            return {
                        'success':True,
                        'function':function,
                        'input': c_id,
                        'output':{
                            'next_action':'x',
                            'message':'No continuity id, Creating new plan'
                        }
                }
                
         
    
    def run_plan_action_tool(self,plan_id,plan_step,action_step,tool_step):
        
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
                
            #pr = f'Executor Trace {execution_result}'
            #print(pr)
            #self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)

            print(json.dumps(execution_result, indent=2, cls=DecimalEncoder))
            
            return {
                'success':True,
                'function': function,
                'input':payload,
                'output':execution_result
            }
            
        except Exception as e:
            pr = f'🤖❌ @rpat:{e}'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=self._get_context().connection_id)
            
            return {
                'success':False,
                'function':function,
                'input':payload,
                'output':e
            }
            
        
        

    
    def run(self,payload):
        

        
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
        
        '''
        
        
        '''
        
            CONTINUITY ROUTER RULES
            If the c_id shows 
            [ p:-:-:- ] Resume from plan_step 0 
            [ p:n:-:- ] Resume from plan_step n
            [ p:n:m:- ] Resume from plan_step n , action_step m
            [ p:m:m:q ] Resume from plan_step n , action_step m , tool_step q 
            
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
        action = 'run > Hierarchical Agent'
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
            pr = f"Analyzing continuity ID..."
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            
            pr = f'Running continuity router for [{context.next}] ...'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            
            response_1 = self.continuity_router(context.next)
            results.append(response_1)
            if not response_1['success']: 
                return {'success':False,'action':action,'output':results}
            
            pr = f'The Continuity Router recommendation is: {response_1['output']} '
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            
            
            '''
            Continuity routes
            
            1. create_plan
            2. initiate_plan
            3. initiate_action
            4. initiate_tool
            5. resume_tool   
            
            '''

            next_action = response_1.get('output', {}).get('next_action', None)
            plan_id = response_1.get('output', {}).get('plan_id', None)
            plan_step = response_1.get('output', {}).get('plan_step', 0)
            action_step = response_1.get('output', {}).get('action_step', 0)
            tool_step = response_1.get('output', {}).get('tool_step', 0)
            
            
            # PLAN GENERATION
            
            if next_action == 'generate_plan':
                pr = 'Initializing the plan generator...'
                print(pr)
                self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
                
                # Load prompts from text files
                prompts = self.prompts
                
                generator = GeneratePlan(prompts=prompts)
                
                req = {
                    "portfolio":context.portfolio,
                    "org":context.org,
                    "message":context.message
                }
                print('Running plan generator...')
                out = generator.run(req)
                
                if not out['success']:
                    raise Exception(out['output'])
                
                plan = out['output'][0]['output']['plan']
                plan_id = plan['id']
                
                # Save the plan in the short term memory
                #self.save_plan(plan) #TO-BE-IMPLEMENTED
                self.AGU.mutate_workspace({'plan': plan}, public_user=self._get_context().public_user, workspace_id=self._get_context().workspace_id)
                next_action = 'initiate_plan'
                
                m = { "role": "assistant", "content":{"plan_id":plan_id}}  
                self.AGU.save_chat(m)
                   
            
            # PLAN EXECUTION
               
            # We pass the plan_id and not the plan itself. Executor will retrieve it from workspace
            pr = f'Calling the executor for plan:{plan_id}, plan_step:{plan_step}, action_step:{action_step}, tool_step:{tool_step}'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            

            if next_action == 'initiate_plan':
                response_1a = self.run_plan_action_tool(plan_id,0,0,0)
            elif next_action == 'initiate_action':
                response_1a = self.run_plan_action_tool(plan_id,plan_step,0,0)
            elif next_action == 'initiate_tool':
                response_1a = self.run_plan_action_tool(plan_id,plan_step,action_step,0)
            elif next_action == 'resume_tool':
                response_1a = self.run_plan_action_tool(plan_id,plan_step,action_step,tool_step)
                
            results.append(response_1a)
                
            execution_results = json.dumps(response_1a, indent=2, cls=DecimalEncoder)
            pr = f'Execution output: {execution_results}'
            print(pr)
            #self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            m = { "role": "assistant", "content":f'🤖🤖'}
            
            self.AGU.save_chat(m, connection_id=context.connection_id)
            #self.AGU.print_chat(f'🤖','text', connection_id=context.connection_id) 
            return {'success':True,'action':action,'input':payload,'output':results}
                    
  
        except Exception as e:
            pr = f'🤖❌:{e}'
            print(pr)
            self.AGU.print_chat(pr,'text', connection_id=context.connection_id)
            return {'success':False,'action':action,'output':f'Run failed. Error:{str(e)}','stack':results}

    

# Test block
if __name__ == '__main__':
    # Creating an instance
    pass
