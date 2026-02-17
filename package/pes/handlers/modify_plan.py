from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union
import json, uuid, re
import os
from decimal import Decimal
from datetime import datetime

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
# LLM client (OpenAI-backed) with task-specific Structured Outputs
# ────────────────────────────────────────────────────────────────────────────────

class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...
    def complete_json(self, prompt: str) -> Dict[str, Any]: ...

class AIResponsesLLM:
    """
    Real OpenAI-backed LLM that supports JSON-structured planning without calling actions.
    It inspects the prompt for a TASK tag and applies an appropriate JSON schema.
    Supports both structured outputs (gpt-4o+) and JSON mode (gpt-3.5-turbo, gpt-4-turbo).
    """
    def __init__(self, agu: AgentUtilities, model: Optional[str] = None):
        # Use the provided AgentUtilities instance instead of creating a new one
        self.AGU = agu
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def _supports_structured_outputs(self) -> bool:
        """Check if the model supports structured outputs (json_schema)."""
        # Models that support structured outputs
        structured_models = ["gpt-4o", "gpt-4o-mini", "gpt-5"]
        return any(self.model.startswith(m) for m in structured_models)

    # Helper: safely extract text from AGU.llm() response
    def _resp_text(self, resp) -> str:
        # AGU.llm() returns message object directly (response.choices[0].message)
        try:
            if hasattr(resp, 'content') and resp.content:
                return resp.content
            return ""
        except Exception:
            return ""

    def _detect_task(self, prompt: str) -> str:
        m = re.search(r"TASK:\s*([A-Z0-9_]+)", prompt)
        return m.group(1) if m else ""

    def _schema_for_task(self, task: str) -> Dict[str, Any]:
        # Schema for plan modification
        if task == "MODIFY_PLAN":
            return {
                "name": "Plan",
                "schema": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "meta": {"type": "object", "additionalProperties": False},
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step_id": {"type": "integer"},
                                            "title": {"type": "string"},
                                            "action": {"type": "string"},
                                            "inputs": {"type": "object", "additionalProperties": False},
                                            "enter_guard": {"type": "string"},
                                            "success_criteria": {"type": "string"},
                                            "depends_on": {
                                                "type": "array",
                                                "items": {"type": "integer"}
                                            },
                                            "next_step": {
                                                "anyOf": [
                                                    {"type": "integer"},
                                                    {"type": "null"}
                                                ]
                                            }
                                        },
                                        "required": ["step_id","title","action","inputs","enter_guard","success_criteria","depends_on"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["id","steps"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["plan"],
                    "additionalProperties": False
                },
                "strict": True
            }
        # Default permissive JSON object
        return {
            "name": "GenericJSON",
            "schema": {"type": "object"},
            "strict": False
        }

    def complete(self, prompt: str) -> str:
        """
        For tasks that expect free-form text or lightweight JSON.
        """
        response_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        resp = self.AGU.llm(response_kwargs)
        return self._resp_text(resp)

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        """
        Returns parsed JSON using Structured Outputs (if available) or JSON mode.
        """
        task = self._detect_task(prompt)
        schema = self._schema_for_task(task)
        
        # Prepare the API call
        if self._supports_structured_outputs():
            # Use structured outputs for newer models
            request_params = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'response_format': {"type": "json_schema", "json_schema": schema}
            }
            resp = self.AGU.llm(request_params)
        else:
            # Use JSON mode for older models (gpt-3.5-turbo, gpt-4-turbo, etc.)
            # Include the schema in the prompt so the model knows the expected structure
            schema_str = json.dumps(schema["schema"], indent=2)
            enhanced_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
            request_params = {
                'model': self.model,
                'messages': [{"role": "user", "content": enhanced_prompt}],
                'response_format': {"type": "json_object"}
            }
            resp = self.AGU.llm(request_params)
        
        txt = self._resp_text(resp)
        try:
            return json.loads(txt)
        except Exception:
            # Best-effort: attempt to find a JSON block in text
            m = re.search(r"\{[\s\S]*\}\s*$", txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            # Fallback empty object
            return {}
        
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
# Modifier
# ────────────────────────────────────────────────────────────────────────────────

class Modifier:
    """
    Plan modification that uses LLM to modify existing plans.
    Action catalog provided to LLM; validator enforces action/arg correctness.
    """
    def __init__(self,
                 llm: LLMClient,
                 action_catalog: List[ActionSpec],
                 prompts: Optional[Dict[str, str]] = None):
        self.llm = llm
        self.action_catalog = action_catalog
        self.prompts = prompts or {}
    
    def _replace_tokens(self, prompt: str, replacements: Dict[str, Any]) -> str:
        """
        Replace tokens in prompt template with actual values.
        Supports #token# format (new) and {token} format (legacy).
        JSON tokens are automatically wrapped in ```json ... ``` code blocks.
        
        Args:
            prompt: Template string with tokens
            replacements: Dictionary mapping token names to values
            
        Returns:
            Prompt with all tokens replaced
        """
        # Tokens that should be wrapped in JSON code blocks
        json_tokens = {
            'intent_text', 'sig_text', 'example_cases', 'fact_texts', 'catalog_summary',
            'catalog', 'skills', 'activities_requested', 'plan_details',
            'existing_plan'
        }
        
        # Special handling for state_machine and trip (they include their own formatting)
        special_tokens = {'state_machine', 'trip'}
        
        for token, value in replacements.items():
            # Handle #token# format (new, preferred)
            token_placeholder = '#' + token + '#'
            if token_placeholder in prompt:
                if token in special_tokens:
                    # state_machine and trip already have their formatting
                    replacement = str(value) if value else ''
                elif token in json_tokens:
                    # Wrap JSON tokens in code blocks
                    replacement = f'```json\n{value}\n```'
                else:
                    # Simple replacement for non-JSON tokens
                    replacement = str(value)
                prompt = prompt.replace(token_placeholder, replacement)
            
            # Legacy support for {token} and {{token}} formats
            legacy_placeholder1 = '{' + token + '}'
            legacy_placeholder2 = '{{' + token + '}}'
            if legacy_placeholder1 in prompt or legacy_placeholder2 in prompt:
                if token in special_tokens:
                    replacement = str(value) if value else ''
                elif token in json_tokens:
                    replacement = f'```json\n{value}\n```'
                else:
                    replacement = str(value)
                prompt = prompt.replace(legacy_placeholder1, replacement)
                prompt = prompt.replace(legacy_placeholder2, replacement)
        
        return prompt

    # Plan modification orchestrator
    def modify(self, existing_plan: Plan, modification_request: str, 
               state_machine: Optional[Dict[str, Any]] = None,
               trip: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Modify an existing plan based on a human-readable paragraph describing the changes.
        
        Args:
            existing_plan: The current plan to modify
            modification_request: Human-readable paragraph describing what changes to make
            state_machine: Optional execution state machine showing which steps have been completed 
            trip: Optional trip document showing what reservations already exist 
            
        Returns:
            Dictionary with success status and modified plan
        """
        function = 'modify'
        print('Initiating plan modification process')
        print(f'Modification request: {modification_request}')
        if state_machine:
            print(f'State machine provided: {len(state_machine.get("steps", []))} steps')
        if trip:
            print(f'Trip document provided')
        
        # Convert existing plan to dict for prompt
        existing_plan_dict = {
            "id": existing_plan.id,
            "meta": existing_plan.meta,
            "steps": [asdict(s) for s in existing_plan.steps]
        }
        
        # Build action catalog summary for reference
        catalog_summary = [{"name": t.key, "required_args": t.required_args, "optional_args": t.optional_args} for t in self.action_catalog]
        
        # Prepare state machine and trip information for prompt
        state_machine_json = json.dumps(state_machine, indent=2, cls=DecimalEncoder) if state_machine else None
        trip_json = json.dumps(trip, indent=2, cls=DecimalEncoder) if trip else None
        
        # Use prompt from database if available, otherwise use default
        prompt_template = self.prompts.get('modify_plan', '')
        if prompt_template:
            # Compute values for tokens
            existing_plan_json = json.dumps(existing_plan_dict, indent=2, cls=DecimalEncoder)
            catalog_summary_json = json.dumps(catalog_summary, indent=2)
            modification_request_text = modification_request
            
            # Replace tokens with computed values
            replacements = {
                'existing_plan': existing_plan_json,
                'existing_plan_id': existing_plan.id,
                'modification_request': modification_request_text,
                'catalog_summary': catalog_summary_json
            }
            
            # Add state machine and trip if provided, otherwise use placeholder
            # Format with JSON code blocks for proper display
            if state_machine_json:
                replacements['state_machine'] = f'STATE MACHINE (execution status):{state_machine_json}'
            else:
                replacements['state_machine'] = ''
            
            if trip_json:
                replacements['trip'] = f'TRIP DOCUMENT (existing reservations):{trip_json}'
            else:
                replacements['trip'] = ''
            
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            # Fallback to default hardcoded prompt
            prompt = f"""
            You are a plan modification agent specialized in modifying trip plans.
            TASK: MODIFY_PLAN
            
            IMPORTANT PRINCIPLES:
            - A plan is a high-level step-by-step list of segments
            - Modifying a trip plan consists of REPLACING high-level segments with new ones, not modifying existing segments
            - Segments are immutable - you can only replace them, not change them
            - Once you replace a segment, a specialist agent will execute it
            - Once a trip plan has been modified, it needs to be approved by the user
            
            EXECUTION STATE RULES (CRITICAL):
            - If a state_machine is provided, it shows which steps have already been executed
            - Check step status in state_machine.steps[step_id].status:
              * "completed" = Step is already executed and cannot be modified directly
              * "pending" or "awaiting" = Step has not been executed and CAN be modified
            - If a trip document is provided, it shows what reservations already exist (flights, hotels, etc.)
            - MODIFICATION CONSTRAINTS FOR COMPLETED STEPS:
              1. You CANNOT modify a step that has status "completed" in the state machine
              2. To change a completed step, you MUST:
                 a) Create a NEW step with action "remove_segment" that references the completed segment in the trip document
                 b) Then create another NEW step with the updated parameters (e.g., "quote_flight" with new parameters)
                 c) Position these new steps AFTER all completed steps in the plan
              3. The "remove_segment" action needs to point to the segment in the trip document:
                 - For flights: reference the flight in trip.flights array (use leg/index)
                 - For hotels: reference the hotel in trip.hotels array (use leg/index)
                 - The remove_segment step removes the existing reservation from the trip document
              4. Example workflow for changing a completed flight:
                 - Original step 0: "quote_flight" EWR→DEN (status: "completed")
                 - User wants: Change to ORD→DEN
                 - Solution: 
                   * Keep original step 0 unchanged (it's completed)
                   * Add new step N (after all completed steps): "remove_segment" pointing to the flight in trip.flights[0]
                   * Add new step N+1: "quote_flight" with ORD→DEN parameters
            - MODIFICATION CONSTRAINTS FOR PENDING STEPS:
              1. You CAN modify steps that have status "pending" or "awaiting" or don't exist in state machine
              2. You CAN add new segments at any time
              3. Always check BOTH state_machine AND trip document to determine what's already executed
              4. Keep the big picture - some changes affect multiple segments (e.g., hotel nights → return flight date)
            
            TYPES OF CHANGES:
            a) Elimination: Remove a segment from the plan (no replacement needed)
               - If step is "pending" or "awaiting": Simply remove it from the plan
               - If step is "completed": Create a "remove_segment" step pointing to the trip document item
            b) Parameter modification: Change parameters of an existing segment (same function, different params)
               - If step is "pending" or "awaiting": Modify the step directly
               - If step is "completed": 
                 * Create a "remove_segment" step pointing to the trip document item
                 * Create a new step with updated parameters
                 * Position both after completed steps
            c) Function modification: Change the function of an existing segment to an equivalent one
               - If step is "pending" or "awaiting": Modify the step directly
               - If step is "completed":
                 * Create a "remove_segment" step pointing to the trip document item
                 * Create a new step with the new function and parameters
                 * Position both after completed steps
            d) Add segment: Add a new segment to the trip
               - Always allowed, regardless of execution state
               - Position after completed steps if there are any
            
            EXISTING PLAN:
            ```json
            {json.dumps(existing_plan_dict, indent=2, cls=DecimalEncoder)}
            ```
            
            {f'''STATE MACHINE (execution status):
            ```json
            {state_machine_json}
            
            ''' if state_machine_json else ''}
            
            {f'''TRIP DOCUMENT (existing reservations):
            ```json
            {trip_json}
            ```
            ''' if trip_json else ''}
            
            MODIFICATION REQUEST:
            {modification_request}
            
            ACTION CATALOG (ensure inputs match required_args):
            ```json
            {json.dumps(catalog_summary, indent=2)}
            ```
            
            INSTRUCTIONS:
            1. **CHECK EXECUTION STATE FIRST** - Before making any changes:
               - Review the state_machine (if provided) to see which steps are "completed", "pending", or "awaiting"
               - Review the trip document (if provided) to see what reservations already exist
               - Count how many steps are "completed" - new steps must be positioned AFTER these
               - Identify which steps can be modified directly vs which need remove_segment + new step
            2. Analyze the modification request carefully to understand what changes are needed
            3. **HANDLE COMPLETED STEPS** - For any completed step that needs changes:
               - DO NOT modify the completed step itself
               - Find the corresponding item in the trip document (e.g., trip.flights[0] for step 0 if it's a flight)
               - Create a new "remove_segment" step with inputs pointing to the trip document item:
                 * For flights: {{"section": "flights", "leg": 0}} (or appropriate index)
                 * For hotels: {{"section": "hotels", "leg": 0}} (or appropriate index)
               - Create a new step with the updated parameters/function
               - Position both new steps AFTER all completed steps
            4. **HANDLE PENDING STEPS** - For pending/awaiting steps:
               - You can modify them directly in place
               - Update their parameters, function, or remove them as needed
            5. Identify which segments need to be:
               - Eliminated (removed) - check if completed first
               - Modified (parameters or function changed) - check if completed first
               - Added (new segments) - always allowed
            6. **CRITICAL: CASCADING CHANGES** - When modifying one segment, check if other segments need updates:
               - Hotel stay duration changes → Recalculate return flight departure_date
               - Hotel check-in date changes → Recalculate return flight if it depends on checkout
               - Arrival flight date changes → Update hotel check-in_date if applicable
               - Adding/removing cities → May require new flights, hotels, or removal of existing segments
               - Always verify temporal consistency: dates must flow logically (arrival → check-in → checkout → departure)
            7. **DATE CALCULATIONS** - When hotel nights change:
               - Check-in date + number_of_nights = Checkout date
               - Return flight departure_date should typically match or follow the checkout date
               - Example: If hotel check-in is 2026-01-30 with 3 nights, checkout is 2026-02-02, so return flight should be on or after 2026-02-02
               - If nights change to 5, checkout becomes 2026-02-04, so return flight must be updated to 2026-02-04 or later
            8. **STEP POSITIONING** - Maintain proper step ordering:
               - Keep all completed steps in their original positions (DO NOT modify them)
               - Place new steps (remove_segment, new quotes, etc.) AFTER completed steps
               - Example: If steps 0,1 are completed and step 2 is pending:
                 * Keep steps 0,1 unchanged
                 * New steps start at position 2 or later
                 * Reorder remaining pending steps as needed
            9. Apply the changes while maintaining logical flow and dependencies
            10. Ensure all step_id values are sequential (0, 1, 2, 3...)
            11. Update depends_on and next_step relationships correctly
            12. Preserve the plan id unless creating a new version
            
            OUTPUT FORMAT:
            Return ONLY JSON with this exact structure:
            {{
                "plan": {{
                    "id": "{existing_plan.id}",
                    "meta": {{"strategy": "modified", "original_id": "{existing_plan.id}"}},
                    "steps": [
                        {{
                            "step_id": 0,
                            "title": "descriptive title",
                            "action": "actionName",
                            "inputs": {{"arg1": "value1", ...}},
                            "enter_guard": "True",
                            "success_criteria": "appropriate criteria",
                            "depends_on": [],
                            "next_step": 1
                        }}
                    ]
                }}
            }}
            
            STEP STRUCTURE REQUIREMENTS:
            - step_id: Sequential integer starting from 0 (0, 1, 2, 3, ...)
            - depends_on: Array of step_id integers that this step depends on
            - next_step: The step_id of the next step in execution order, or null for the last step
            - title: Clear, descriptive title for the step
            - action: Must match an action name from the ACTION CATALOG
            - inputs: Object with all required_args from the action spec, plus any optional_args
            - enter_guard: Boolean expression (typically "True")
            - success_criteria: Expression to evaluate step success
            
            CHRONOLOGICAL ORDER (CRITICAL):
            - Steps MUST be listed in the order they occur in time
            - step_id values MUST be sequential: 0, 1, 2, 3...
            - next_step links to the following step (step 0 → 1 → 2 → 3 → null)
            - Ensure dependencies are satisfied (depends_on steps execute before dependent steps)
            
            FINAL VALIDATION:
            - Does the modified plan address ALL requirements from the modification request?
            - Are all step dependencies valid?
            - Are all actions valid and have required arguments?
            - Is the plan logically complete and executable?
            - **EXECUTION STATE CHECK**: Have you respected the execution state constraints?
              - Did you avoid modifying steps with status "completed"?
              - Did you create overwrite steps for completed reservations that need changes?
              - Did you check both state_machine and trip document to determine execution state?
            - **TEMPORAL CONSISTENCY CHECK**: Are all dates logically consistent?
              - Hotel check-in dates align with arrival flights
              - Hotel checkout dates (check-in + nights) align with departure flights
              - All dates flow chronologically (earlier steps happen before later steps)
            - **CASCADING CHANGES CHECK**: Have all affected segments been updated?
              - If hotel nights changed, is the return flight date updated?
              - If arrival date changed, are hotel check-in dates updated?
              - If a segment was removed, are dependent segments updated or removed?
            """
        
        print(f'modify LLM Prompt >> {prompt}')
        data = self.llm.complete_json(prompt)
        if not data or "plan" not in data:
            print('[ERROR] LLM returned invalid plan structure')
            return {
                'success': False,
                'function': function,
                'input': {'existing_plan': existing_plan_dict, 'modification_request': modification_request},
                'output': 'LLM returned invalid plan structure'
            }
        
        # Create steps
        steps = []
        for idx, s in enumerate(data["plan"]["steps"]):
            if "action" not in s or not s.get("action"):
                print(f'[WARNING] Step {idx} missing or empty action field! Step data: {s}')
            # Ensure all required fields are present
            if "step_id" not in s:
                s["step_id"] = len(steps)  # Fallback to sequential
            if "depends_on" not in s:
                s["depends_on"] = []
            if "next_step" not in s:
                s["next_step"] = None
            try:
                steps.append(PlanStep(**s))
            except Exception as e:
                print(f'[ERROR] Failed to create PlanStep from step {idx}: {e}')
                print(f'[ERROR] Step data: {json.dumps(s, indent=2, cls=DecimalEncoder)}')
                continue
        
        plan = Plan(
            id=data["plan"].get("id", existing_plan.id),
            steps=steps,
            meta=data["plan"].get("meta", {"strategy": "modified", "original_id": existing_plan.id})
        )
        
        print('Raw modified plan:')
        print(json.dumps({"id": plan.id, "meta": plan.meta, "steps": [asdict(s) for s in plan.steps]}, indent=2, cls=DecimalEncoder))
        
        validated_plan = self._validate_and_patch_plan(plan)
        
        return {
            "function": function,
            "success": True,
            "input": {
                "existing_plan": existing_plan_dict,
                "modification_request": modification_request
            },
            "output": {
                "plan": asdict(validated_plan)
            }
        }

    # Validator: ensure known actions + required args are present
    def _validate_and_patch_plan(self, plan: Plan) -> Plan:
        print(f'Validating plan with {len(plan.steps)} steps...')
        known = {t.key: t for t in self.action_catalog}
        validated: List[PlanStep] = []
        for s in plan.steps:
            print(f'  Validating step {s.step_id}: action={repr(s.action)}, action_type={type(s.action)}, inputs={s.inputs}')
            
            # Check if action is empty or None
            if not s.action or s.action == "":
                print(f'    REJECTED: empty or missing action field (value: {repr(s.action)})')
                continue
            
            # Check if action is in catalog
            if s.action not in known:
                print(f'    REJECTED: unknown action "{s.action}"')
                continue
            
            spec = known[s.action]
            inputs = s.inputs or {}
            
            missing = [a for a in spec.required_args if a not in inputs]
            if missing:
                print(f'    REJECTED: missing required args {missing}')
                provided_keys = list(inputs.keys()) if inputs else []
                continue
            
            # Populate optional fields if not present
            if not s.success_criteria and spec.success_criteria_hint:
                s.success_criteria = spec.success_criteria_hint
            if not s.enter_guard:
                s.enter_guard = "True"
            print(f'    ACCEPTED')
            validated.append(s)
        print(f'Validation complete: {len(validated)}/{len(plan.steps)} steps kept')
        plan.steps = validated
        return plan



# ────────────────────────────────────────────────────────────────────────────────
# Handler implementation
# ────────────────────────────────────────────────────────────────────────────────


# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

'''
This class handles modification of existing trip plans.
It takes an existing plan and a human-readable paragraph describing modifications,
then uses LLM-based plan modification to produce an updated plan.
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
        Returns a dictionary with keys: 'to_intent', 'adapt_plan', 'compose_plan', 'select_best_plan', 'modify_plan'
        """
        prompts = {
            'to_intent': '',
            'adapt_plan': '',
            'compose_plan': '',
            'select_best_plan': '',
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
                    elif key == 'adapt_plan':
                        prompts['adapt_plan'] = prompt_text
                    elif key == 'compose_plan':
                        prompts['compose_plan'] = prompt_text
                    elif key == 'select_best_plan':
                        prompts['select_best_plan'] = prompt_text
                    elif key == 'modify_plan':
                        prompts['modify_plan'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts from database: {str(e)}')
            
        
        return prompts
    
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
    
    
    def build_plan_modifier(self, portfolio: str, org: str, 
                         prompt_ring: str = "pes_prompts",
                         action_ring: str = "schd_actions",
                         plan_actions: Optional[List[str]] = None) -> Modifier:
        """
        Build a Modifier instance configured for plan modification.

        Loads prompts and actions from the database. If plan_actions is provided,
        the action catalog is filtered to only those actions (so the LLM and
        validator only see allowed actions for this request).
        
        Returns:
            Modifier: Configured agent ready for plan modification operations
        """
        # Pass the AgentUtilities instance to AIResponsesLLM
        llm = AIResponsesLLM(self.AGU)

        # Get case_group from context
        context = self._get_context()
        case_group = context.case_group if context else None
        
        # Use prompts from initialization if available, otherwise load from database
        if self.prompts:
            prompts = self.prompts
        else:
            prompts = self._load_prompts(portfolio, org, prompt_ring, case_group=case_group)
        
        # Load actions from database
        action_catalog = self._load_actions(portfolio, org, action_ring)
        
        # Filter to allowed plan_actions when provided (from payload _init)
        if plan_actions:
            allowed = set(plan_actions)
            action_catalog = [a for a in action_catalog if a.key in allowed]
        
        # Note: If no actions are loaded from database, action_catalog will be empty
        if not action_catalog:
            print('Warning: No actions loaded from database, action_catalog will be empty')

        # Note: We don't load seed cases or facts since modify() doesn't use VDB retrieval
        # The modify() method works directly with the existing plan and modification request

        return Modifier(llm=llm, action_catalog=action_catalog, prompts=prompts)
    
    
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
            
            # Create AgentUtilities with throw away entity_type, entity_id and thread 
            # The reason is that this class doesn't write to the chat entity directly
            # We are only initializing AGU because we need access to its LLM function
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
            modifier = self.build_plan_modifier(
                portfolio=context.portfolio,
                org=context.org,
                prompt_ring="pes_prompts",  # Can be made configurable
                action_ring="schd_actions",  # Can be made configurable
                plan_actions=plan_actions
            )
            print('Finished building Planner')
            
            # Parse existing plan from payload
            existing_plan_data = payload['plan']
            if isinstance(existing_plan_data, str):
                plan_str = existing_plan_data.strip()
                print(f'[DEBUG] Parsing plan string (length: {len(plan_str)}, first 100 chars: {plan_str[:100]})')
                
                # Find the first complete JSON object by tracking braces
                brace_count = 0
                start_pos = -1
                end_pos = -1
                for i, char in enumerate(plan_str):
                    if char == '{':
                        if start_pos == -1:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos != -1:
                            end_pos = i + 1
                            break
                
                if start_pos != -1 and end_pos != -1:
                    # Extract just the JSON object part
                    json_str = plan_str[start_pos:end_pos]
                    print(f'[DEBUG] Extracted JSON (length: {len(json_str)}, first 100 chars: {json_str[:100]})')
                    try:
                        existing_plan_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f'[DEBUG] JSON decode error: {str(e)}')
                        print(f'[DEBUG] JSON string: {json_str[:500]}')
                        return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_plan/run: Failed to parse plan JSON: {str(e)}'}
                else:
                    # Fallback: try parsing the whole string
                    try:
                        existing_plan_data = json.loads(plan_str)
                    except json.JSONDecodeError as e:
                        return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_plan/run: Could not find complete JSON object. Error: {str(e)}'}
            
            # Convert plan dict to Plan object
            existing_steps = [PlanStep(**s) for s in existing_plan_data.get('steps', [])]
            existing_plan = Plan(
                id=existing_plan_data.get('id', uuid.uuid4().hex[:8]),
                steps=existing_steps,
                meta=existing_plan_data.get('meta', {})
            )
            
            # Get modification request paragraph
            modification_request = payload['message']
            
            # Parse optional state_machine and trip documents
            
            state_machine = None
            if 'state_machine' in payload and payload['state_machine']:
                state_machine_data = payload['state_machine']
                if isinstance(state_machine_data, str):
                    try:
                        state_machine = json.loads(state_machine_data)
                    except json.JSONDecodeError as e:
                        print(f'[WARNING] Could not parse state_machine JSON: {str(e)}')
                        state_machine = None
                elif isinstance(state_machine_data, dict):
                    state_machine = state_machine_data
            
            trip = None
            if 'trip' in payload and payload['trip']:
                trip_data = payload['trip']
                if isinstance(trip_data, str):
                    try:
                        trip = json.loads(trip_data)
                    except json.JSONDecodeError as e:
                        print(f'[WARNING] Could not parse trip JSON: {str(e)}')
                        trip = None
                elif isinstance(trip_data, dict):
                    trip = trip_data
            
            # Call modify method
            response_1 = modifier.modify(existing_plan, modification_request, 
                                        state_machine=state_machine, trip=trip)
            results.append(response_1)
            if not response_1['success']: 
                return {'success': False, 'output': results}
            
            # All went well, report back
            canonical = results[-1]['output']  # This is a dictionary {'plan':{}}
            return {'success': True, 'interface': 'plan', 'input': payload, 'output': canonical, 'stack': results}
            
        except Exception as e:
            print(f'Error during execution: {str(e)}')
            import traceback
            traceback.print_exc()
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@modify_plan/run: {str(e)}'}

