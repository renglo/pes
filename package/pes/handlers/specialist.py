#
from renglo.data.data_controller import DataController
from renglo.schd.schd_controller import SchdController
from renglo.common import load_config

from openai import OpenAI
from datetime import datetime
from typing import List, Dict, Any
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
import json
import random

class UniversalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal, datetime, date, and other common types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'isoformat'):  # Handles date, time, and other datetime-like objects
            return obj.isoformat()
        if hasattr(obj, '__dict__'):  # Handle custom objects by converting to dict
            return obj.__dict__
        return super(UniversalEncoder, self).default(obj)

# Backwards compatibility alias
DecimalEncoder = UniversalEncoder

def sanitize(obj):
    """
    Recursively sanitize an object to ensure it's JSON-serializable.
    Converts Decimal to int/float, datetime to ISO string, etc.
    """
    if obj is None:
        return None
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, 'isoformat') and not isinstance(obj, str):  # date, time, etc.
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(item) for item in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, '__dict__'):
        return sanitize(obj.__dict__)
    # Fallback: convert to string
    return str(obj)

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
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    list_actions: List[Dict[str, Any]] = field(default_factory=list)
    list_tools: List[Dict[str, Any]] = field(default_factory=list)
    continuity: Dict[str, Any] = field(default_factory=dict)
    message: str = ''
    tool_response_c_id: str = ''

# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

class Specialist:
    
    def __init__(self,agu):
        """
        :agu: Agent Utilities
        """
        self.AGU = agu  
        self.config = load_config()
        self.DAC = DataController(config=self.config)
        self.SHC = SchdController(config=self.config)
        
        
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
        
    
    
    def consent_form(self,payload):
        function = 'consent_form'
        
        tool_name = payload['tool_calls'][0]['function']['name']
        arguments = payload['tool_calls'][0]['function']['arguments']
        if isinstance(arguments, str):
            arguments_dict = json.loads(arguments)
        else:
            arguments_dict = arguments
        params = ', '.join([f"{k}: {v}" for k, v in arguments_dict.items()])
        
        consent = {
            'commands':payload['tool_calls'],
            'interface':'binary_consent',
            'nonce': random.randint(100000, 999999),
            'message':{
                "role": "assistant",
                "content": f'I would like to call {tool_name} tool with the following parameters:{params}. Please confirm it is ok'
            }
        }
        
        return consent
        
        

        
    def interpret(self,no_tools=False,tool_result=False):
        
        action = 'interpret'
        self.AGU.print_chat('Interpreting message...', 'transient')
        print('interpret')
        
        try:
            
            
            # The goal of this specialist
            # Belief (coming from the Plan)
            current_beliefs = self._get_context().inputs    
            belief_str = 'Current beliefs: ' + self.AGU.string_from_object(current_beliefs) 
            print(f'Current Beliefs:{belief_str}')
            # Desire (coming from the Plan)
            current_desire = self._get_context().title
            print(f'Current Desire:{current_desire}')
            
            
            # We get the message history directly from the source of truth to avoid missing tool id calls. 
            continuity = self._get_context().continuity
            print(continuity)
            message_filter = {'param':'_next','begins_with':f'irn:c_id:{continuity["plan_id"]}:{continuity["plan_step"]}'}
            message_list = self.AGU.get_message_history(filter=message_filter)
            
            print(f'Specialist Message History: {message_list}')
            
            #If the message_list comes back empty, that means the specialist execution is new. Create into message
            if not message_list['output']:
                if isinstance(current_beliefs, dict):
                    inputs = ', '.join([f"{k}: {v}" for k, v in current_beliefs.items()])
                else:
                    inputs = f'{current_beliefs}'
                    
                step_number = int(continuity["plan_step"])
                intro_msg = {'role':'assistant','content':f'Initiating step {step_number}. {current_desire} with the following parameters: {inputs}'}
                c_id = f'irn:c_id:{continuity["plan_id"]}:{continuity["plan_step"]}'
                self.AGU.save_chat(intro_msg, next = c_id) 
                #message_list['output'].append({'_type':'text','_next':c_id,'_out':intro_msg})
                message_list['output'].append(intro_msg)
            
            # Go through the message_list and replace the value of the 'content' attribute with an empty object when the role is 'tool'
            # Unless the last message it a tool response which the interpret function needs to process. 
            # The reason is that we don't want to overwhelm the LLM with the contents of the history of tool outputs. 
            
            # Clear content from all tool messages except the last one
            message_list = self.AGU.clear_tool_message_content(message_list['output'])
            
            print(f'Cleared Message History')
            
            
            # Get current time and date
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            
            action_instructions = '' 
            action_tools = ''
            list_actions = self._get_context().list_actions
            
            for a in list_actions:
                if a['key'] == self._get_context().current_action:
                    action_instructions = a['prompt_3_reasoning_and_planning']
                    
                    if 'tools_reference' in a and a['tools_reference'] and a['tools_reference'] not in ['_','-','.']: 
                        action_tools = a['tools_reference']
                    break
                
            # Optimal Path
            optimal_path_instructions = (
                "You have an optimal_path with numbered steps. At every turn, you must:\n"
                "Decide which step you are currently executing (current_step_id).\n"
                "Optionally choose the next step you want to move to (next_step_id).\n"
                "Optionally call tools or ask the user.\n"
                "The optimal path is a guide, not a strict pipeline. You may:\n"
                "- Repeat a step (e.g., go from step 1 back to step 1),\n"
                "- Go back to previous steps (e.g., from step 3 to step 1) if you need to refine parameters."    
            )
            
            
            
            # Meta Instructions
            meta_instructions = {}
            # Initial instructions
            meta_instructions['opening_message'] = "You are an AI assistant. You can reason over conversation history, beliefs, and goals."
            # Provide the current time
            meta_instructions['current_time'] = f'The current time is: {current_time}'
            # Message to answer questions from the belief system
            meta_instructions['answer_from_belief'] = "You can reason over the message history and known facts (beliefs) to answer user questions. If the user asks a question, check the history or beliefs before asking again."
                  
            # Message array
            messages = [
                { "role": "system", "content": meta_instructions['opening_message']}, # META INSTRUCTIONS
                { "role": "system", "content": meta_instructions['current_time']}, # CURRENT TIME         
                { "role": "system", "content": action_instructions}, # CURRENT ACTIONS
                { "role": "system", "content": optimal_path_instructions}, # OPTIMAL PATH INSTRUCTIONS
                { "role": "system", "content": meta_instructions['answer_from_belief']},
                { "role": "system", "content": belief_str }, # BELIEF SYSTEM
                { "role": "system", "content": current_desire }, # CURRENT_DESIRE
                
            ]
            
            # Add the incoming messages
            for msg in message_list:      
                messages.append(msg)       
                
            # Initialize approved_tools with default empty list
            approved_tools = []
                
            # Request asking the recommended tools for this action
            if action_tools and not no_tools:
                messages.append({ "role": "system", "content":f'In case you need them, the following tools are recommended to execute this action: {json.dumps(action_tools)}'})  
                
                approved_tools = [tool.strip() for tool in action_tools.split(',')]
                    
            # Tools           
            '''   
            tool.input should look like this in the database:
                
                {
                    "origin": { 
                        "type": "string",
                        "description": "The departure city code or name",
                        "required":true
                    },
                    "destination": { 
                        "type": "string", 
                        "description": "The arrival city code or name",
                        "required":true
                    }
                }
            '''
            
            
            if no_tools:                
                list_tools = None      
                   
            else:         
                list_tools_raw = self._get_context().list_tools
                
                #print(f'List Tools:{list_tools_raw}')
                
                list_tools = [] 
                for t in list_tools_raw:
                    
                    if t.get('key') in approved_tools:
                        # Parse the escaped JSON string into a Python object
                        try:
                            tool_input = json.loads(t.get('input', '[]'))
                        except json.JSONDecodeError:
                            print(f"Invalid JSON in tool input for tool {t.get('key', 'unknown')}. Using empty array.")
                            tool_input = []
                        
                        dict_params = {}
                        required_params = []
                        
                        # Handle new format: array of objects with name, hint, required
                        if isinstance(tool_input, list):
                            for param in tool_input:
                                if isinstance(param, dict) and 'name' in param and 'hint' in param:
                                    param_name = param['name']
                                    param_hint = param['hint']
                                    param_required = param.get('required', False)
                                    
                                    dict_params[param_name] = {
                                        'type': 'string',
                                        'description': param_hint
                                    }
                                    
                                    if param_required:
                                        required_params.append(param_name)
                        # Handle old format for backward compatibility
                        elif isinstance(tool_input, dict):
                            for key, val in tool_input.items():
                                dict_params[key] = {'type': 'string', 'description': val}
                                required_params.append(key)
                                
                        print(f'Required parameters:{required_params}')
                            
                        tool = {
                            'type': 'function',
                            'function': {
                                'name': t.get('key', ''),
                                'description': t.get('goal', ''),
                                'parameters': {
                                    'type': 'object',
                                    'properties': dict_params,
                                    'required': required_params
                                }
                            }    
                        }
                        
                        #print(f'Tool:{tool}')       
                        list_tools.append(tool)          
                        #print(f'List Tools:{list_tools}')
                    
                    
            # Prompt
            prompt = {
                    "model": self.AGU.AI_1_MODEL,
                    "messages": messages,
                    "tools": list_tools,
                    "temperature":0,
                    "tool_choice": "auto"
                }
            
            
            prompt = self.AGU.sanitize(prompt)
            
            #print(f'RAW PROMPT >> {prompt}')
            response = self.AGU.llm(prompt)
            print(f'RAW RESPONSE >> {response}')
          
            
            if not response:
                return {
                    'success': False,
                    'action': action,
                    'input': '',
                    'output': response
                }
                
            
            validation = self.AGU.validate_interpret_openai_llm_response(response)
            if not validation['success']:
                return {
                    'success': False,
                    'action': 'validation',
                    'input': response,
                    'output': validation
                }
            
            validated_result = validation['output']
            
            
            # We infer the action_step by analyzing the validated_result.
            # Some steps in the optimal path have a tool, some don't
            # The same tool could be used in multiple steps. 
            # We could use the message roll to infer what step are on. But the payroll is kind of saturated with other actions. It could be confusing
            # We could report every action loop to the state machine. Based on that history, we could compare it with the official optimal path to find the current state
            # We could use an auxiliary LLM call to ask what action_step are we on, based on all the data available. (but it would be better to do it programmatically)
            
            
            continuity = self._get_context().continuity
            c_id_pre = f'irn:c_id:{continuity["plan_id"]}:{continuity["plan_step"]}' 
            
            
            if 'role' in validated_result:
            
                if validated_result.get('tool_calls') and validated_result.get('role') == 'assistant':
                    # This is the LLM asking for a tool to be executed.
                    selected_tool = validated_result['tool_calls'][0]['function']['name']
                    
                    if (continuity['tool_step'] == '3' or continuity['tool_step'] == '4' ) and continuity['action_step'] == selected_tool :
                        print(f'Interpret() >> Run this tool:{validated_result}')
                        # The last message was a response to "3 = WAITING HUMAN".
                        # The continuity response matches with the selected_tool. Execute tool
                        
                        # We are running the tool
                        nonce = random.randint(100000, 999999)
                        tool_step = '4' # 4  = EXECUTION_REQUEST      # tool execution has been requested by agent
                        c_id = f'{c_id_pre}:{selected_tool}:{tool_step}:{nonce}' 
                        self.AGU.save_chat(validated_result, next = c_id)           
                         
                        log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "tool":selected_tool,
                            "status":tool_step,
                            "nonce":nonce,
                            "message":"Consent provided, executing tool",
                            "type":"consent_ok"
                        }
                        self.AGU.mutate_workspace({'action_log': log_entry})
                        
                    else:
                        
                        print(f'Interpret() >> Switching tool call into a message: {validated_result}')
                        # We are turning the tool call into a message to the user
                        
                        tool_step = 3 # 3  = WAITING_HUMAN   waiting for human confirmation / input
                        consent = self.consent_form(validated_result)
                        c_id = f'{c_id_pre}:{selected_tool}:{tool_step}:{consent["nonce"]}' 
                        self.AGU.save_chat(consent['message'], msg_type='consent', next = c_id) 
                        
                        validated_result = consent['message'] # Replacing original message with consent message.         
                        
                        # Recording it in the action_log
                        log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "tool":selected_tool,
                            "status":tool_step,
                            "nonce":consent["nonce"],
                            "message":consent["message"]["content"],
                            "type":"consent_rq"
                        }
                        self.AGU.mutate_workspace({"action_log": log_entry})
                        
                    
                        
                elif validated_result.get('role') == 'assistant':
                    
                    if tool_result == 'tool_error':
                        # This is the interpretation of the error message by the tool
                        print(f'Interpret() >> The agent has interpreted the tool error:{validated_result}')
                        msg = validated_result.get('content')
                        
                        self.AGU.save_chat(validated_result)
                            
                        log_entry = {
                                "plan_id":continuity["plan_id"],
                                "plan_step":continuity["plan_step"],
                                "message":msg      
                        } 
                        
                        self.AGU.mutate_workspace({"action_log": log_entry}) 
                        
                        
                    else:
                    # This is the LLM asking something to the user. 
                        if tool_result == 'fresh_results':
                            print(f'Interpret() >> The agent has interpreted the tool results:{validated_result}')
                            c_id = self._get_context().tool_response_c_id
                            c_id_parts = c_id.split(':')
                            nonce = c_id_parts[6]
                            msg = validated_result.get('content')
                            #f'irn:c_id:{continuity["plan_id"]}:{continuity["plan_step"]}:*:3:{nonce}'
                        else:
                            print(f'Interpret() >> The agent is asking something to the user: {validated_result}')
                            nonce = random.randint(100000, 999999)
                            c_id = f'{c_id_pre}:*:1:{nonce}'
                            msg = validated_result.get('content')
                            
                            self.AGU.save_chat(validated_result, next=c_id)
                            
                            log_entry = {
                                    "plan_id":continuity["plan_id"],
                                    "plan_step":continuity["plan_step"],
                                    "tool":"*",
                                    "status":"0",
                                    "nonce":nonce,
                                    "message":msg,
                                    "type":"decision_rq"
                                    
                            } 
                            self.AGU.mutate_workspace({"action_log": log_entry}) 
                      
            return {
                'success': True,
                'action': action,
                'input': '',
                'output': validated_result
            }
            
        except Exception as e:
            print(f"Error in interpret() message: {e}")
            return {
                'success': False,
                'action': action,
                'input': '',
                'output': str(e)
            }
    
        
        
    ## Execution of Intentions
    def act(self,command):
        function = 'act'
        
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
    
        """Execute the current intention and return standardized response"""
        try:
            
            tool_name = command['tool_calls'][0]['function']['name']
            params = command['tool_calls'][0]['function']['arguments']
            if isinstance(params, str):
                params = json.loads(params)
            tid = command['tool_calls'][0]['id']
            
            print(f'tid:{tid}')

            if not tool_name:
                raise ValueError("❌ No tool name provided in tool selection")
                
            print(f"Selected tool: {tool_name}")
            self.AGU.print_chat(f'Calling tool {tool_name} with parameters {params} ', 'transient')
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
            
            # Check if init exists and is valid
            handler_init = {}
            if not isinstance(list_inits[tool_name], str) and isinstance(list_inits[tool_name], dict):
                handler_init = list_inits[tool_name]
                
                
            # Check if handler has the right format (2 parts: tool/handler, or 3 parts: tool/handler/subhandler)
            handler_route = list_handlers[tool_name]
            parts = handler_route.split('/')
            if len(parts) < 2 or len(parts) > 3:
                error_msg = f"❌ {tool_name} is not a valid tool. Handler route must be 'tool/handler' or 'tool/handler/subhandler'."
                print(error_msg)
                self.AGU.print_chat(error_msg, 'error')
                raise ValueError(error_msg)
            
            # For 3-part routes (tool/handler/subhandler), combine handler and subhandler
            tool = parts[0]
            handler = '/'.join(parts[1:])  # "handler" or "handler/subhandler"

            portfolio = self._get_context().portfolio
            org = self._get_context().org
            
            params['_portfolio'] = self._get_context().portfolio
            params['_org'] = self._get_context().org
            params['_entity_type'] = self._get_context().entity_type
            params['_entity_id'] = self._get_context().entity_id
            params['_thread'] = self._get_context().thread
            params['_init'] = handler_init
            
            print(f'Calling {handler_route} ') 
            
            response = self.SHC.handler_call(portfolio, org, tool, handler, params)
            
            #response = {'success':True,'output':{"some":"mockup response"}}
            
            print(f'Handler response:{response}')
            
            #raise Exception('Troubleshooting stop')

            if not response['success']:
                # The handler came back with an error. The output contains the handler's execution list
                raise Exception (response['output'])

            # The response of every handler always comes in 'output'
            clean_output = sanitize(response['output'])
            clean_output_str = json.dumps(clean_output)
            
            interface = None
            # The handler determines the interface
            if 'interface' in response:
                interface = response['interface']
            print(f'@act:Interface:{interface}')
            
            tool_out = {
                    "role": "tool",
                    "tool_call_id": f'{tid}',
                    "content": clean_output_str,
                    "tool_calls":False
                }
            
            # Custom assembling the c_id to become the consent_form for the next step which is for the user to take action on the tool results.
            # The asterisk (*) means any tool that is going to process the response to this results.
            continuity = self._get_context().continuity
            nonce = random.randint(100000, 999999)
            c_id = f'irn:c_id:{continuity["plan_id"]}:{continuity["plan_step"]}:{tool_name}:5:{nonce}'

            # Save the message after it's created
            # Important: Because we had to create a response message in advance, we are upserting an existing message, not creating a new one. 
            #  The upsert only takes 'content', '_interface' and '_next' changes. 
            
            self.AGU.save_chat(tool_out,interface=interface, next=c_id)     
            
            
            print(f'act:Saved tool results to chat')
            
            # Results coming from the handler
            self._update_context(tool_response_c_id=c_id)
            
            print(f'act:Saved tool response c_id to context')
            
            # Save handler result to workspace
            
            # Turn an object like this one: {"people":"4","time":"16:00","date":"2025-06-04"}
            # Into a string like this one: "4/16:00/2026-06-04"
            # If the value of each key is not a string just output an empty space in its place
            #params_str = self.format_object_to_slash_string(params)
            index = f'irn:tool_rs:{handler_route}' 
            tool_input = command['tool_calls'][0]['function']['arguments'] 
            #input is a serialize json, you need to turn it into a python object before inserting it into the value dictionary
            tool_input_obj = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            value = {'input': tool_input_obj, 'output': clean_output}
            
            
            #Reporting to the State Machine
            log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "tool":tool_name, # The tool that will process this response
                            "status":"5", # This 3 refers to the 
                            "nonce":nonce,
                            "message":"Tool executed.",
                            "type":"tool_ok"
                        }
            
            
            self.AGU.mutate_workspace(
                {
                    'cache': {index:value},
                    'action_log': log_entry
                }, 
                workspace_id=self._get_context().workspace_id
            )
            
            print(f'flag5')
            
            #print(f'message output: {tool_out}')
            print("✅ Tool execution complete.")
            
            
            return {"success": True, "function": function, "input": command, "output": tool_out}
                    
        except Exception as e:
            
            # Notice that this exception won't leave a trace in the messages. 
            # Interpret uses messages to read the output of act(). 
            # We are passing the error results via the context instead

            error_msg = f"❌ Tool failed. Trying something different. @act trying to run tool:'{tool_name}': {str(e)}"
            self.AGU.print_chat(error_msg,'error') 
            self._update_context(execute_intention_error=error_msg)
            
            continuity = self._get_context().continuity
            log_entry = {
                'plan_id':continuity['plan_id'],
                'plan_step':continuity['plan_step'],
                'message': f'Tool failed:{e}'
            }
            self.AGU.mutate_workspace({'action_log': log_entry})           
            
            error_result = {
                "success": False, "action": action,"input": command,"output": str(e)    
            }
            
            
            return error_result   
        
    
    def check(self,command):
        function = 'check'
        tool_name = None  # Initialize to avoid UnboundLocalError
        
        try: 
            
            list_tools_raw = self._get_context().list_tools
            
            list_handlers = {}
            for t in list_tools_raw:
                list_handlers[t.get('key', '')] = t.get('handler', '')
                
            tool_name = command['tool_calls'][0]['function']['name']
            params = command['tool_calls'][0]['function']['arguments']
            
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
            

            response = self.SHC.handler_check(portfolio,org,parts[0],parts[1],params)

            return {"success": True, "action": function, "input": "", "output": response}
              
              
        except Exception as e:
            
            tool_name_str = tool_name if tool_name else 'unknown'
            error_msg = f"❌ Check failed. @check :'{tool_name_str}': {str(e)}"
            self.AGU.print_chat(error_msg,'error') 
            print(error_msg)
            self._update_context(execute_intention_error=error_msg)
            
            continuity = self._get_context().continuity
            
            error_result = {
                "success": False, "action": function,"input": command,"output": str(e)    
            }
            
            log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "tool":tool_name_str,
                            "nonce":random.randint(100000, 999999),
                            "message":"Check failed"
                        }
            self.AGU.mutate_workspace({'action_log': log_entry})
            
            return error_result  
        
        
    def verify(self,action):
        function = 'verify'
        verification_handler = None  # Initialize to avoid UnboundLocalError
        params = {}  # Initialize params dictionary
        
        try: 
            
            for a in self._get_context().list_actions:
                if a['key'] == self._get_context().current_action:
                    print(f'@verify:Current Action:{a}')
                    if 'verification' not in a:
                        raise Exception (f'No verification handler found for this action: {a.get("key", "N/A")}')
                            
                    verification_tool = a['verification']
            
            for t in self._get_context().list_tools:
                if t['key'] == verification_tool:
                    verification_handler = t.get('handler', '')
             
            parts = verification_handler.split('/')
            
            if len(parts) != 2:
                error_msg = f"❌ {verification_handler} is not a valid verification tool."
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
            
            continuity = self._get_context().continuity
            params['plan_id'] = continuity["plan_id"]
            params['plan_step'] = continuity["plan_step"]
            
            # Plan and State Machine
            workspace = self.AGU.get_active_workspace()
            params['plan'] = workspace['plan'][continuity['plan_id']]
            params['state_machine'] = workspace['state_machine'][continuity['plan_id']]

            response = self.SHC.handler_call(portfolio,org,parts[0],parts[1],params)
            
            if response['success']:              
                msg = f"Verification OK. Step Completed."
                tool_step = '5'
                continuity = self._get_context().continuity  
                log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "status":tool_step,
                            "message":msg
                }
                self.AGU.mutate_workspace({'action_log': log_entry})
                
                self.AGU.print_chat(msg,'transient') 
                print(msg)
            
            else:     
                msg = f"Step has not been completed yet. Continue the loop"
                continuity = self._get_context().continuity  
                log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "message":msg
                }
                self.AGU.mutate_workspace({'action_log': log_entry})
                # We don't need to record in action_log when verification is KO 
                print(msg) 
            
    
            return {"success": response['success'], "action": function, "input": "", "output": response['output']}
                 
              
        except Exception as e:
            
            verification_handler_str = verification_handler if verification_handler else 'unknown'
            error_msg = f"❌ Verification failed. @verify :'{verification_handler_str}': {str(e)}"
            self.AGU.print_chat(error_msg,'error') 
            print(error_msg)
            self._update_context(execute_intention_error=error_msg)
            
            continuity = self._get_context().continuity
            
            error_result = {
                "success": False, "action": function,"input": action,"output": str(e)    
            }
            
            log_entry = {
                            "plan_id":continuity["plan_id"],
                            "plan_step":continuity["plan_step"],
                            "tool":verification_handler_str,
                            "status":"6",
                            "nonce":random.randint(100000, 999999),
                            "message":f'Verification failed: {e}'
                        }
            self.AGU.mutate_workspace({'action_log': log_entry})
            
            return error_result  
            
             
    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
   
        
    def run(self, payload):
        
        '''
        The payload contains the plan step 
        
        EXAMPLE STEP
        {
            "action": "quote_flight",
            "depends_on": [],
            "enter_guard": "True",
            "inputs": {
                "from_airport_code": "MIA",
                "outbound_date": "2025-12-14",
                "passengers": "3",
                "to_airport_code": "LAS"
            },
            "next_step": "1",
            "step_id": "0",
            "success_criteria": "len(result) > 0",
            "title": "Miami to Las Vegas outbound flight"
            "continuity":{
                "plan_id":"",
                "plan_step":"",
                "action_step":"",
                "tool_step":""
            } #Added in the executor
            
        }
        '''
        
        '''
        tool_step (status code)
        0  = DEFAULT                # Default, tool has not being started
        1  = REQUEST_INFO           # Agent requesting information from human
        2  = WAITING_TOOL_IO        # waiting for external tool response / callback
        3  = WAITING_HUMAN          # waiting for human confirmation / input
        4  = EXECUTION_REQUEST      # tool execution has been requested by agent
        5  = COMPLETED_OK           # tool finished
        6  = COMPLETED_ERROR        # tool failed / exception / unrecoverable
        7  = CANCELLED              # step aborted (by user/system)
        '''
        
        action = 'run > specialist'
        print(f'Running specialist:{payload}')
        
        # Get context from AGU if available, otherwise create new
        
        try:
            
            context = RequestContext()
            
            # Populate context from AGU
            if hasattr(self.AGU, 'portfolio'):
                context.portfolio = self.AGU.portfolio
                context.org = self.AGU.org
                context.entity_type = self.AGU.entity_type
                context.entity_id = self.AGU.entity_id
                context.thread = self.AGU.thread
            
            # Get available actions and tools
            actions = self.DAC.get_a_b(context.portfolio, context.org, 'schd_actions')
            context.list_actions = actions['items']
            
            tools = self.DAC.get_a_b(context.portfolio, context.org, 'schd_tools')
            context.list_tools = tools['items']
            
            
            # Step information
            context.inputs = payload.get('inputs',{})
            context.title = payload.get('title','')
            
            context.step_id = payload.get('step_id','')
            context.current_action = payload.get('action', '')
            context.continuity = payload.get('continuity',{}) # plan_id, plan_step, action_step, tool_id

            
            # Set the initial context for this turn
            self._set_context(context)
            
            results = []
            
            #A. Extract the action from the step object. 
            if not context.current_action:
                return {'success': False, 'action': action, 'input': payload, 'output': 'No action specified in step'}
            # Save Action document. The specialist will follow it.
            self.AGU.mutate_workspace({'action': context.current_action})
                    
            # Run the ReAct loop

            loops = 0
            loop_limit = 6
            tool_result = ''
            verification_result = ''
            while loops < loop_limit:
                loops = loops + 1
                print(f'Loop iteration {loops}/{loop_limit}')
                
                
                # Step 1: Interpret. We receive the message from the user and we issue a tool command or another message       
                response_1 = self.interpret(tool_result=tool_result)
                tool_result = ''
                
                print(f'Run() >> Response from interpret: {response_1}')
                
                results.append(response_1)
                if not response_1['success']:
                    # Something went wrong during message interpretation
                    print('Something went wrong during interpret(). Exiting specialist')
                    return {'success':False,'action':action,'output':response_1,'stack':results}  
                
                
                if verification_result:
                    output = {
                        'status':'completed'
                    }
                    return {'success':True,'action':action,'input':payload, 'output':output ,'stack':results}
                       
                
                # Check whether we need to run a tool
                
                if 'tool_calls' in response_1['output'] and response_1['output']['tool_calls']: 
                    # Tool need Execution
                    # Step 2: Act. Agent runs the tool

                    print(f'Run() >> Tool Execution:{response_1["output"]}') 
                    response_2 = self.act(response_1['output'])
                    results.append(response_2)
                    tool_result = 'fresh_results'
                    
                        
                    if not response_2['success']:
                        #print('Tool failed, feeding tool output to loop. Agent will try to fix it. Otherwise will exit.')
                        # Something went wrong during tool execution, Have the agent try to fix it instead of just giving up.
                        #return {'success':False,'action':action,'output':response_2,'stack':results}
                        tool_result = 'tool_error'
                        
                        
                        
                        #continue
                                      
                    
                    '''
                    # Tool returned successfully. Run tool custom checks
                    response_2b = self.check(response_2['output'])
                    results.append(response_2b)
                    if not response_2b['success']:
                        tool_result = 'tool_error'
                    '''
                        
                        
                    # Run verification script to figure out if action is done
                    response_2c= self.verify(context.current_action)
                    results.append(response_2c)
                    if response_2c['success']:
                        # Action is done, exit the specialist
                        verification_result = 'action_done'
                        # Instead or returning here, we'll let the agent interpret the tool results first
                        # but we won't give the agent the opportunity to call another tool. 
                        
                        log_entry = {
                            'plan_id':context.continuity['plan_id'],
                            'plan_step':context.continuity['plan_step'],
                            'message':'Step completed'
                        }
                        
                        self.AGU.mutate_workspace({'action_log': log_entry})
                        
                        
                        
                elif 'tool_calls' not in response_1['output'] or not response_1['output']['tool_calls']:
                    # No Tool needs execution. 
                    # Most likely the agent is asking for more information to fill tool parameters. 
                    # Or agent is answering questions directly from the belief system.
                    print(f'Run() >> Specialist exits because it sent a direct message to the user.')
                    
                    self.AGU.print_chat(f'🤖','transient')
                    
                    # The above code is creating a Python dictionary named `output` with a single
                    # key-value pair. The key is 'status' and the value is 'awaiting'.
                    
                    output = {
                        'status':'awaiting'
                    }
                    
                    return {'success':True,'action':action,'input':payload, 'output':output ,'stack':results}
                                
                           
                                  
            
            #Gracious exit. Analyze the last tool run (act()) but you can't issue a new tool_call. 
            response_3 = self.interpret(no_tools=True)
            results.append(response_3)
            if not response_3['success']:
                    # Something went wrong during message interpretation
                    return {'success':False,'action':action,'output':response_3,'stack':results} 
            
            
            # If we reach here, we hit the loop limit
            print(f'Warning: Reached maximum loop limit ({loop_limit})')
            self.print_chat(f'🤖⚠️  Can you re-formulate your request please?','text')
            return {'success':True,'action':action,'input':payload,'output':response_3['output'],'stack':results}
            
        
        except Exception as e: 
            self.AGU.print_chat(f'🤖❌(Specialist):{e}','transient')
            return {'success':False,'action':action,'output':f'Run failed. Error:{str(e)}','stack':results}

        
        
        

        
        
