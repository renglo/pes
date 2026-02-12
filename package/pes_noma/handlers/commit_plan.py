# get_available_documents.py
from renglo.data.data_controller import DataController
from renglo.docs.docs_controller import DocsController
from renglo.auth.auth_controller import AuthController
from renglo.chat.chat_controller import ChatController
from renglo.blueprint.blueprint_controller import BlueprintController
from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Any
from decimal import Decimal
#from jsonschema import validate, ValidationError
from openai import OpenAI


import json
import re

# Custom JSON encoder to handle Decimal objects
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

@dataclass
class RequestContext:
    portfolio: str = ''
    org: str = ''
    entity_type: str = ''
    entity_id: str = ''
    thread: str = ''
    init: Dict[str, Any] = field(default_factory=dict)
    search_results: Dict[str, Any] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)

# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

'''
This class searches for documents of a specific type
'''

class CommitPlan:
    def __init__(self):
        # Load config for handlers (independent of Flask)
        self.config = load_config()
        
        #OpenAI Client
        try:    
            openai_api_key = self.config.get('OPENAI_API_KEY', '')
            openai_client = OpenAI(api_key=openai_api_key)
            print(f"OpenAI client initialized")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            openai_client = None

        self.AI_1 = openai_client
        #self.AI_1_MODEL = "gpt-4" // This model does not support json_object response format
        self.AI_1_MODEL = "gpt-3.5-turbo" # Baseline model. Good for multi-step chats
        self.AI_2_MODEL = "gpt-4o-mini" # This model is not very smart
        
        # Initialize controllers with config
        self.DAC = DataController(config=self.config)
        self.AUC = AuthController(config=self.config)
        self.DCC = DocsController(config=self.config)
        self.BPC = BlueprintController(config=self.config)
        self.CHC = ChatController(config=self.config)
        

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
    
    def sanitize(self, obj):
        """
        Recursively convert Decimal objects to regular numbers in nested data structures.
        """
        if isinstance(obj, list):
            return [self.sanitize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self.sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, Decimal):
            # Convert Decimal to int if it's a whole number, otherwise float
            return int(obj) if obj % 1 == 0 else float(obj)
        else:
            return obj
    
    '''
    def validate_hotel_segment(self, segment):
        """
        Validates a hotel segment against the VacationRental JSON schema.
        
        Args:
            segment (dict): The hotel segment to validate
            
        Returns:
            dict: Contains 'success' (bool), 'output' (dict), and 'message' (str) if validation fails
        """
        # VacationRental JSON Schema
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "VacationRental",
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "format": "uri"
                },
                "amenities": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "thumbnail": {
                    "type": "string",
                    "format": "uri"
                },
                "distance": {
                    "type": "string"
                },
                "reviewScore": {
                    "type": "string"
                },
                "originalPrice": {
                    "type": "string"
                },
                "adults": {
                    "type": "string"
                },
                "latitude": {
                    "type": "string"
                },
                "rating": {
                    "type": "string"
                },
                "currentPrice": {
                    "type": "string"
                },
                "discount": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                },
                "availability": {
                    "type": "string"
                },
                "reviewCount": {
                    "type": "integer"
                },
                "nights": {
                    "type": "integer"
                },
                "name": {
                    "type": "string"
                },
                "id": {
                    "type": "string"
                },
                "reviewText": {
                    "type": "string"
                },
                "roomType": {
                    "type": "string"
                },
                "longitude": {
                    "type": "string"
                }
            },
            "required": ["image", "amenities", "thumbnail", "latitude", "longitude", "currentPrice", "name", "id", "roomType"]
        }
        
        
        if isinstance(segment, str):
            try:
                print(f"Attempting to parse JSON string: {segment[:200]}...")  # Show first 200 chars
                
                # Clean the JSON string first
                cleaned_json = self._clean_json_string(segment)
                print(f"Cleaned JSON string: {cleaned_json[:200]}...")
                
                parsed_data = json.loads(cleaned_json)
                print(f"JSON parsed successfully, type: {type(parsed_data).__name__}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"Error position: line {e.lineno}, column {e.colno}")
                print(f"Error message: {e.msg}")
                return {
                    'success': False,
                    'input':segment,
                    'message': f"Invalid JSON string: {str(e)} at line {e.lineno}, column {e.colno}",
                    'output': None
                }
        elif isinstance(segment, dict):
            parsed_data = segment
        else:
            return {
                'success': False,
                'input':segment,
                'message': f"Payload must be a dictionary or JSON string, got {type(segment).__name__}",
                'output': None
            }      
        
        try:
         
            # Validate the segment against the schema
            validate(instance=parsed_data, schema=schema)
            
            # If validation passes, return success with the validated segment
            return {
                'success': True,
                'output': parsed_data
            }
            
        except ValidationError as e:
            # If validation fails, return error with details
            return {
                'success': False,
                'output': None,
                'message': f"Validation failed: {str(e)}"
            }
        except Exception as e:
            # Handle any other unexpected errors
            return {
                'success': False,
                'output': None,
                'message': f"Unexpected error during validation: {str(e)}"
            }
        
    '''
    
        
    def _clean_json_string(self, json_str):
        """Clean common JSON formatting issues"""
        import re
        
        # Remove trailing commas before closing braces and brackets
        # This regex finds commas followed by closing braces/brackets and removes the comma
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix Python boolean values to JSON boolean values
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        
        # Fix None to null
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        
        return json_str
    
    
    
    def clean_json_response(self, response):
        """
        Cleans and validates a JSON response string from LLM.
        
        Args:
            response (str): The raw JSON response string from LLM
            
        Returns:
            dict: The parsed JSON object if successful
            None: If parsing fails
            
        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON
        """
        try:
            # Clean the response by ensuring property names are properly quoted
            #cleaned_response = response.strip() 
            cleaned_response = response
            # Remove any comments (both single-line and multi-line)
            cleaned_response = re.sub(r'//.*?$', '', cleaned_response, flags=re.MULTILINE)  # Remove single-line comments
            cleaned_response = re.sub(r'/\*.*?\*/', '', cleaned_response, flags=re.DOTALL)  # Remove multi-line comments
            
            # First try to parse as is
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                pass
                
            # If that fails, try to fix common issues
            # Handle unquoted property names at the start of the object
            cleaned_response = re.sub(r'^\s*{\s*(\w+)(\s*:)', r'{"\1"\2', cleaned_response)
            
            # Handle unquoted property names after commas
            cleaned_response = re.sub(r',\s*(\w+)(\s*:)', r',"\1"\2', cleaned_response)
            
            # Handle unquoted property names after newlines
            cleaned_response = re.sub(r'\n\s*(\w+)(\s*:)', r'\n"\1"\2', cleaned_response)
            
            # Replace single quotes with double quotes for property names
            cleaned_response = re.sub(r'([{,]\s*)\'(\w+)\'(\s*:)', r'\1"\2"\3', cleaned_response)
            
            # Replace single quotes with double quotes for string values
            # This regex looks for : 'value' pattern and replaces it with : "value"
            cleaned_response = re.sub(r':\s*\'([^\']*)\'', r': "\1"', cleaned_response)
            
            # Remove spaces between colons and boolean values
            cleaned_response = re.sub(r':\s+(true|false|True|False)', r':\1', cleaned_response)
            
            # Remove trailing commas in objects and arrays
            # This regex will match a comma followed by whitespace and then a closing brace or bracket
            cleaned_response = re.sub(r',(\s*[}\]])', r'\1', cleaned_response)
            
            # Remove any timestamps in square brackets
            cleaned_response = re.sub(r'\[\d+\]\s*', '', cleaned_response)
            
            # Try to parse the cleaned response
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"First attempt failed. Error: {e}")
                #print(f"Cleaned response type: {type(cleaned_response)}")
                #print(f"Cleaned response length: {len(cleaned_response)}")
                #print(f"Cleaned response content: '{cleaned_response}'")
                
                # If first attempt fails, try to fix the raw field specifically
                # Find the raw field and ensure it's properly formatted
                raw_match = re.search(r'"raw":\s*({[^}]+})', cleaned_response)
                if raw_match:
                    raw_content = raw_match.group(1)
                    # Convert single quotes to double quotes in the raw content
                    raw_content = raw_content.replace("'", '"')
                    # Replace the raw field with the cleaned version
                    cleaned_response = cleaned_response[:raw_match.start(1)] + raw_content + cleaned_response[raw_match.end(1):]
                
                #print(f"After raw field cleanup - content: '{cleaned_response}'")
                return json.loads(cleaned_response)
        
                
        except json.JSONDecodeError as e:
            print(f"Error parsing cleaned JSON response: {e}")
            #print(f"Original response: {response}")
            #print(f"Cleaned response: {cleaned_response}")
            raise
        
        
            
    def llm(self, prompt):
          
        try:
            
            # Create base parameters
            params = {
                'model': prompt['model'],
                'messages': prompt['messages'],
                'temperature': prompt['temperature']
            }
        
            # Add optional parameters if they exist
            if 'tools' in prompt:
                params['tools'] = prompt['tools']
            if 'tool_choice' in prompt:
                params['tool_choice'] = prompt['tool_choice']
                
            response = self.AI_1.chat.completions.create(**params) 
            
            # chat.completions.create might return an error if you include Decimal() as values
            # Object of type Decimal is not JSON serializable
            
            return response.choices[0].message
 
        
        except Exception as e:
            print(f"Error running LLM call: {e}")
            # Only print raw response if it exists 
            return False
            
    def find_in_cache(self):
        
        action = 'find_in_cache'
        
        try:
            portfolio = self._get_context().portfolio
            org = self._get_context().org
            
            entity_type = self._get_context().entity_type
            entity_id = self._get_context().entity_id
            thread = self._get_context().thread
            init = self._get_context().init
            
            
                                
            # Get the workspaces in this thread 
            response = self.CHC.list_workspaces(portfolio,org,entity_type,entity_id,thread) 
            workspaces_list = response['items']
            print('WORKSPACES_LIST >>',workspaces_list)
            
            if not workspaces_list or len(workspaces_list) == 0:
                print('No workspaces found')
                return {
                    'success': False,
                    'action': action,
                    'error': 'No workspaces found for this thread',
                    'output': 0
                }
            
            # Extract cache from workspace
            workspace = workspaces_list[0]
            if 'cache' not in workspace:
                print('No cache found in workspace')
                return {
                    'success': False,
                    'action': action,
                    'error': 'No cache found in workspace',
                    'output': 0
                }
            
            cache_keys = []
            if 'plan_cache_key' in init:
                cache_keys.append(f'irn:tool_rs:{init["plan_cache_key"]}')
            else:    
                cache_keys.append('irn:tool_rs:pes_noma/modify_plan')
                cache_keys.append('irn:tool_rs:pes_noma/generate_plan')
                
            for cache_key in cache_keys:
                entry = workspace['cache'].get(cache_key)
                if not entry:
                    continue
                output = entry.get('output') or {}
                plan = output.get('plan')
                if plan is not None:
                    print('Plan:', plan)
                    return {'success': True, 'action': action, 'input': '', 'output': plan}
                
            print(f'Cache key {cache_key} not found')
            return {
                'success': False,
                'action': action,
                'error': f'Plan with cache key {cache_key} not found in workspace',
                'output': 0
            }
            
        
                
        except Exception as e:
            print(f'Error in find_in_cache: {str(e)}')
            return {
                'success': False,
                'action': action,
                'error': f'Error in find_in_cache: {str(e)}',
                'output': 0
            }
            
     
     
            
    def add_plan(self,plan):
        function = 'add_plan'
        
        try:
            pr = f'add_plan > plan: {plan}'
            print(pr)
            saved = self.AGU.mutate_workspace({'plan': plan})
            plan_id = plan['id']
            
            if saved:
                return {'success':True,'function':function,'input': plan,'output':plan_id}
            else:
                return {'success':False,'function':function,'input': plan,'output':plan_id}
            
        except Exception as e:
            pr = f'Error in saving plan: {str(e)}'
            print(pr)
            return {
                'success': False,
                'function': function,
                'error': pr,
                'output': 0
            }
    
            
            
    def run(self, payload):
        action = 'run>commit_plan'
        
        '''
        INPUT PAYLOAD
        {
            "_portfolio": "string",
            "_org": "string",
            "_entity_type": "org-trip",
            "_entity_id": "string",
            "_thread": "string",
            "case_group": "string",
            "portfolio": "string",
            "org": "string",
            "tool": "string",
        }
        '''
        
        # Initialize a new request context
        context = RequestContext()
        
        # Update context with payload data
        if '_portfolio' in payload or 'portfolio' in payload:
            context.portfolio = payload.get('_portfolio', payload.get('portfolio'))
        else:
            return {'success':False,'action':action,'input':payload,'output':'No portfolio provided'}
        
        if '_org' in payload or 'org' in payload:
            context.org = payload.get('_org', payload.get('org'))
        else:
            context.org = '_all' #If no org is provided, we switch the Agent to portfolio level

        if '_entity_type' in payload:
            context.entity_type = payload['_entity_type']
        else:
            return {'success':False,'action':action,'input':payload,'output':'No entity_type provided'}
        
        if '_entity_id' in payload:
            context.entity_id = payload['_entity_id']
        else:
            return {'success':False,'action':action,'input':payload,'output':'No entity_id provided'}
        
        if '_thread' in payload:
            context.thread = payload['_thread']
        else:
            return {'success':False,'action':action,'input':payload,'output':'No thread provided'}
        
        if '_init' in payload:
            context.init = payload['_init']
        else:
            context.init = {}
                
        self._set_context(context)
        
        
        self.AGU = AgentUtilities(
            self.config,
            context.portfolio,
            context.org,
            context.entity_type,
            context.entity_id,
            context.thread
        )
        
        
        results = []
        
        response_1 = self.find_in_cache()
        results.append(response_1)
        if not response_1['success']: 
            return {'success': False, 'output': results}
        
        response_2 = self.add_plan(response_1['output'])
        results.append(response_2)
        if not response_2['success']: 
            return {'success': False, 'output': results}
        
        output = {
            'next_action':'initiate_plan',
            'plan_id':response_2['output'],
            'plan_step':0
            } 
        # All went well, report back
        return {'success': True, 'interface':'', 'input': payload, 'output': output ,'stack': results}

# Test block
if __name__ == '__main__':
    # Creating an instance
    pass

    
    
    
