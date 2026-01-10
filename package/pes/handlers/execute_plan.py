import json
import time
import copy
from typing import Any, Callable, Dict, List, Optional
from decimal import Decimal
from .specialist import Specialist


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)


class ExecutePlan:
    """
    Executes a plan where each step goes through:
      1. Input Check
      2. Guard Check
      3. Call Action
      4. Check Success Criteria
      5. Return Summary at the end

    Plan progress is accumulated in a JSON-serializable object stored
    in the workspace.
    """

    def __init__(
        self,
        agu,
    ):
        """
        :agu: Agent Utilities
        """
        self.AGU = agu

    # ---------- Internal helpers ----------

    def _init_plan_state(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initial plan execution document that will be stored in the DB.
        """
        now = self._now()
        plan_state = {
            "plan_id": plan["id"],
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "context": {
                # Shared blackboard for step results, intermediate variables, etc.
                "step_execution": {}
            },
            "steps": [],
        }

        for step in plan["steps"]:
            plan_state["steps"].append(
                {
                    "step_id": step.get("step_id"),
                    "title": step.get("title"),
                    "status": "pending",  # pending | running | success | failed | skipped | blocked
                    "result": None,
                    "error": None,
                    "started_at": None,
                    "finished_at": None,
                }
            )

        # Put initialized states in state machine
        self.AGU.mutate_workspace({"new_state_machine": plan_state})
        return plan_state

    def _dependencies_satisfied(
        self, step: Dict[str, Any], step_states_by_id: Dict[str, Dict[str, Any]]
    ) -> bool:
        deps = step.get("depends_on") or []
        if not deps:
            return True
        # All dependencies must be in a terminal state (success/failed/skipped/blocked)
        # "awaiting" is not a terminal state - step is paused waiting for user input
        for dep_id in deps:
            dep_state = step_states_by_id.get(dep_id)
            if not dep_state or dep_state["status"] in (
                "pending",
                "running",
                "awaiting",
            ):
                return False
        return True

    def _dependencies_failed_or_blocked(
        self, step: Dict[str, Any], step_states_by_id: Dict[str, Dict[str, Any]]
    ) -> bool:
        deps = step.get("depends_on") or []
        for dep_id in deps:
            dep_state = step_states_by_id.get(dep_id)
            if dep_state and dep_state["status"] in ("failed", "blocked"):
                return True
        return False

    def _input_check(self, step: Dict[str, Any]) -> None:
        """
        Stage 1: Basic input validation.

        Override / extend this method if you want richer validation
        (schemas, required fields, types, etc.).
        """
        inputs = step.get("inputs")
        if inputs is None:
            raise ValueError(f"Step {step['step_id']} is missing 'inputs'")
        if not isinstance(inputs, dict):
            raise TypeError(f"Step {step['step_id']} 'inputs' must be a dict")
        # Example: ensure inputs are not empty
        if len(inputs) == 0:
            raise ValueError(f"Step {step['step_id']} has empty 'inputs'")

    def _guard_check(self, step: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Stage 2: Evaluate the enter_guard expression (string).
        If it evaluates to False, the step is skipped.
        """
        expr = step.get("enter_guard", "True")
        if expr is None:
            return True

        context = state.get("context", {})

        # Very small "safe" eval environment
        safe_globals = {"__builtins__": None}
        safe_locals = {
            "context": context,
            "True": True,
            "False": False,
        }

        try:
            value = eval(expr, safe_globals, safe_locals)
            return bool(value)
        except Exception as e:
            # If guard expression is broken, we fail the step upstream
            raise ValueError(f"Guard check failed for step {step['step_id']}: {e}")

    def _call_specialist(self, payload: Dict[str, Any]) -> Any:
        """
        Stage 3: Call the action registered under step["action"].

        :param payload: The step definition and continuity variables
        """
        function = '_call_specialist'
        try:
            action_name = payload.get("action")
            if not action_name:

                return {
                    "success": False,
                    "function": function,
                    "input": payload,
                    "output": f"Step {payload['step_id']} has no 'action' specified",
                }
                
            print("Caller payload:", payload)
            
            '''
            payload example
            {
                "depends_on": [],
                "success_criteria": "len(result) > 0",
                "next_step": 1,
                "inputs": {
                    "to_airport_code": "MIA",
                    "passengers": 4,
                    "from_airport_code": "EWR",
                    "outbound_date": "2026-01-01",
                },
                "action": "quote_flight",
                "enter_guard": "True",
                "step_id": 0,
                "title": "Newark to Miami outbound flight",
                "continuity": {"plan_id": "c37333bc","plan_step": 0, "action_step": 0, "tool_step": 0},
            }
            '''
            
            caller = Specialist(self.AGU)
            response = caller.run(payload)
            # Sanitize the response to convert any exception objects to strings
            #print("Caller response:", response)
            
            response = self._sanitize(response)

            if not response["success"]:
                print("Specialist came back with an error:",response)
            
            
            '''
            When step is complete, this comes back in the response
            response['output'] -> {"status": "completed"}
            '''
            
            return response

        except Exception as e:
            pr = f"🤖❌ @_call_specialist:{e}"
            print(pr)
            self.AGU.print_chat(pr, "error")

            return {
                "success": False,
                "function": function,
                "input": payload,
                "output": self._sanitize(e),
            }

    def _check_success(
        self, step: Dict[str, Any], result: Any, state: Dict[str, Any]
    ) -> bool:
        """
        Stage 4: Evaluate success_criteria expression, if provided.
        Expression has access to:
          - result
          - context
          - a few safe builtins (len, any, all, min, max)
        """
        expr = step.get("success_criteria")
        if not expr:
            # If no success criteria is defined, assume success
            return True

        context = state.get("context", {})

        safe_globals = {
            "__builtins__": None,
            "len": len,
            "any": any,
            "all": all,
            "min": min,
            "max": max,
        }
        safe_locals = {
            "result": result,
            "context": context,
            "True": True,
            "False": False,
        }

        try:
            value = eval(expr, safe_globals, safe_locals)
            return bool(value)
        except Exception as e:
            raise ValueError(
                f"Success criteria failed to evaluate for step {step['step_id']}: {e}"
            )

    def _sanitize(self, obj):
        """
        Recursively convert Decimal objects and exception objects to regular numbers/strings
        in nested data structures. This prevents JSON serialization errors.
        """
        if isinstance(obj, list):
            return [self._sanitize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, Decimal):
            # Convert Decimal to int if it's a whole number, otherwise float
            return int(obj) if obj % 1 == 0 else float(obj)
        elif isinstance(obj, BaseException):
            # Convert exception objects to string representation
            return f"{type(obj).__name__}: {str(obj)}"
        else:
            return obj

    def _build_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: Build a compact summary of execution.
        This is what you'll usually return to the caller.
        """
        summary = {
            "plan_id": state["plan_id"],
            "status": state["status"],
            "steps": [
                {
                    "step_id": s["step_id"],
                    "title": s.get("title"),
                    "status": s["status"],
                    "error": s["error"],
                    # you can omit results or keep only key fields if big
                    "result": s["result"],
                    "started_at": s["started_at"],
                    "finished_at": s["finished_at"],
                }
                for s in state["steps"]
            ],
            "context": state.get("context", {}),
        }
        # Sanitize the entire summary to convert any Decimal values
        return self._sanitize(summary)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


    # ---------- Public API ----------

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:

        function = "execute_plan > run"
        print("Running:", function, payload)

        """
        Execute (or resume) the given plan.

        :param payload: Dictionary with the continuity parameters
        :return: A summary dict with overall status and per-step info.
        """

        try:

            plan_id = payload["plan_id"]

            # Continuity object
            continuity = {}
            continuity["plan_id"] = payload["plan_id"]
            # Normalize plan_step to string for consistent comparison with step_id
            continuity["plan_step"] = str(payload["plan_step"])
            continuity["action_step"] = payload.get("action_step", '')
            continuity["tool_step"] = str(payload.get("tool_step", 0))

            workspace = self.AGU.get_active_workspace()
            print("Retrieving plan from active workspace:", workspace)

            # Validate workspace structure
            if "plan" not in workspace:
                raise KeyError("Workspace missing 'plan' key")
            if plan_id not in workspace["plan"]:
                raise KeyError(f"Plan ID '{plan_id}' not found in workspace")

            plan = workspace["plan"][plan_id]

            # Safely access state_machine
            if "state_machine" in workspace and plan_id in workspace["state_machine"]:
                plan_state = workspace["state_machine"][plan_id]
            else:
                plan_state = None

            if plan_state is None:
                plan_state = self._init_plan_state(plan)
                
            print(f'Plan Steps:{plan["steps"]}') 
            steps_by_id = {str(step["step_id"]): step for step in plan["steps"]}
            print(f'PlanState Steps:{plan_state["steps"]}') 
            step_states_by_id = {str(s["step_id"]): s for s in plan_state["steps"]}
            

            
            print("Starting plan execution...")
            loop = 0
            # This is the plan_steps loop
            for step in plan["steps"]:
                loop = loop+1
                
                step_id = str(step["step_id"]) 
                step_title = step["title"]

                pr = f"@ step {step_id}:{step_title}"
                print(pr)
                self.AGU.print_chat(pr, "transient")
                
                if loop > 1:
                    # If this is not the initial loop, the state machine will be stale.
                    # Refreshing local state machine
                    workspace = self.AGU.get_active_workspace()
                    plan_state = workspace['state_machine'][plan_id]
                    step_states_by_id = {str(s["step_id"]): s for s in plan_state["steps"]}
                    # Also, we update the plan step to be current
                    continuity["plan_step"] = step_id
                    
                else:
                    
                    # Compare current step_id to the suggestion in continuity id. If they match, let them start
                    # If they don't match, notify that the continuity id is invalid and that the active step will be initiated. 
                    
                    if step_id != continuity["plan_step"]:
                        pr = "The continuity id is pointing to a completed or non existing step. Proceeding to the active step."
                        print(pr)
                        self.AGU.print_chat(pr, "transient")
                        continuity["plan_step"] = step_id
                    else:
                        pr = "The continuity id is pointing to a valid step."
                        print(pr)
                        self.AGU.print_chat(pr, "transient")
                        
                
                # Ensure step_state exists, break if it doesn't
                if step_id not in step_states_by_id:
                    pr = f"⚠️ Step {step_id} not found in step_states_by_id"
                    print(pr)
                    self.AGU.print_chat(pr, "transient")
                    raise KeyError(f"Step ID '{step_id}' not found in State Machine")
                
                
                # Check if the step has a status that needs to be skipped
                step_state = step_states_by_id[step_id]
                if step_state["status"] in (
                    "completed"
                ):
                    print(f'Skipping step. Status:{step_state["status"]}')
                    continue
                 

                 
                # If the last step completed successfully, it would have declared that step status as completed.
                # When the new loop starts, the executor should advance to the next step naturally without the need of a special signal or flag from the last step. 
                # This is to make the loops independent from execution threads. 
                # If a step_id doesn't exist in the state machine, loop should be aborted. The steps in the state machine were initialized before entering the loop.
                # The only reason to skip a step is if it has been completed. Skipping steps with errors could cause a domino effect. Step dependency checks should take care of that. 
                # A step that is waiting for consent or more data, should have status=awaiting. That status is set by the specialist not by the executor. 
                

                '''
                # Respect dependencies
                print("Checking step dependencies...")
                if not self._dependencies_satisfied(step, step_states_by_id):
                    # At least one dependency failed or is blocked -> block this step
                    if self._dependencies_failed_or_blocked(step, step_states_by_id):
                        print('Step is blocked')
                        
                        # Report dependency error to state machine
                        step_status = {
                            'plan_id':plan_id,
                            'plan_step':step_id,
                            'finished_at':self._now(),
                            'error':'Blocked due to failed dependency',
                            'status':'blocked'
                        }
                        self.AGU.mutate_workspace({"step_state": step_status}) # Changes status of the current step
                        # Step is blocked, skip to next step
                        continue
                        
                    # If dependencies are just not ready yet, skip in this pass
                    msg = "Skipping step because of dependencies..."
                    print(msg)
                    continue
                '''
                  
                # Calling the specialist
                try: 
                  
                    # Report current step status to 'running' to state machine.
                    step_status = {
                        'plan_id':plan_id,
                        'plan_step':step_id,
                        'started_at':self._now(),
                        'status':'running'
                    } 
                    self.AGU.mutate_workspace({"step_state": step_status}) # Changes status of the current step
                    
                    
                    # 1) Input Check
                    self._input_check(step)

                    '''# 2) Guard Check
                    if not self._guard_check(step, plan_state):
                        p = {}
                        p["plan_id"]= plan_id
                        p["plan_step"] = step_id
                        p["status"] = "guarded"
                        p["finished_at"] = self._now()
                        # Step is guarded. Report step as guarded to state machine.
                        self.AGU.mutate_workspace({"step_state": p}) # Changes status of the current step
                        
                        continue
                    '''

                    # 3) Call Action
                    pr = f'Calling action:{step["action"]}'
                    print(pr)
                    self.AGU.print_chat(pr, "transient")

                    specialist_payload = step.copy()
                    specialist_payload["plan_id"] = plan_id

                    # Send continuity variables to the specialist
                    specialist_payload["continuity"] = continuity

                    result = self._call_specialist(specialist_payload)
                    pr = f"Result after calling specialist:{result}"
                    #print(pr)
                    
                    # Recording output status
                    # It could be either 'completed', 'awaiting' or 'error'
                    status = ''
                    if not result.get('success', False):
                        # Specialist returned failure
                        status = 'failed'
                        step_status = {
                            'plan_id': plan_id,
                            'plan_step': step_id,
                            'finished_at': self._now(),
                            'status': 'failed',
                            'error': result.get('output', 'Specialist returned failure')
                        }
                        self.AGU.mutate_workspace({"step_state": step_status})
                        print(f'Specialist returned failure for step {plan_id}:{step_id}')
                        # Continue to next step (or break if you want to stop on failure)
                        break
                    
                    elif result.get('success') and isinstance(result.get('output'), dict) and 'status' in result['output']:
                        status = result['output']['status']
                        
                        step_status = {
                            'plan_id':plan_id,
                            'plan_step':step_id,
                            'finished_at':self._now(),
                            'status':status
                        }     
                        self.AGU.mutate_workspace({"step_state": step_status})
                        
                        print(f'The specialist has declared that step {plan_id}:{step_id} is {status}')
                    
                    
                    
                    if status == 'awaiting':
                        print(f'The step has status {status}.')   
                        # Breaking the loop to wait for answer from user. Loop will be regenerated with the Continuity id.
                        break

                    elif status == 'completed':
                        print('Step has been finished, going to the next step in the plan')
                           

                except Exception as e:
                    # Update step state and persist to state machine
                    step_status = {
                        'plan_id': plan_id,
                        'plan_step': step_id,
                        'status': 'failed',
                        'finished_at': self._now(),
                        'error': f"{type(e).__name__}: {e}"
                    }
                    self.AGU.mutate_workspace({"step_state": step_status})
                    # Also update local reference for consistency
                    step_state = step_states_by_id.get(step_id, {})
                    step_state["status"] = "failed"
                    step_state["finished_at"] = step_status["finished_at"]
                    step_state["error"] = step_status["error"]

                

            # Determine overall status
            pr = f"The execution loop has been suspended. Waiting for further action"
            print(pr)
            #self.AGU.print_chat(pr, "transient")

            '''p = {}
            p["plan_id"] = plan_id
            p["status"] = "completed"
            p["updated_at"] = self._now()
            # Report plan state after all steps have been run or skipped to the state machine
            # Get fresh copy before final write (specialist may have updated state machine during execution)
            self.AGU.mutate_workspace({"plan_state": p}) # Change the status of the plan
            '''

            # 5) Return summary
            '''
            workspace = self.AGU.get_active_workspace()
            plan_state = workspace['state_machine'][plan_id]
            summary = self._build_summary(plan_state)
            '''
            

            return {
                "success": True,
                "function": function,
                "input": payload,
                "output": '',
            }

        except Exception as e:

            pr = f"🤖❌ @execute_plan/run:{e}"
            print(pr)
            self.AGU.print_chat(pr, "error")

            return {
                "success": False,
                "function": function,
                "input": payload,
                "output": e,
            }


# ---------- Example usage with your plan ----------

if __name__ == "__main__":

    # ------- Example: extracting the inner plan from your JSON -------

    plan = {
        "id": "c06e79c8",
        "steps": [
            {
                "step_id": 0,
                "title": "Flight from Rio de Janeiro to New York City",
                "action": "searchFlights",
                "inputs": {
                    "from_airport_code": "GIG",
                    "to_airport_code": "JFK",
                    "outbound_date": "2024-11-15",
                    "passengers": 3,
                },
                "enter_guard": "True",
                "success_criteria": "len(result) > 0 and result[0].get('flight')",
                "depends_on": [],
                "next_step": 1,
            },
            {
                "step_id": 1,
                "title": "Transfer from JFK Airport to Hotel in New York City",
                "action": "searchTransfers",
                "inputs": {"airport": "JFK", "hotel": "Business district"},
                "enter_guard": "True",
                "success_criteria": "result.get('confirmed') == True",
                "depends_on": [0],
                "next_step": 2,
            },
            {
                "step_id": 2,
                "title": "Stay at Hotel in New York City",
                "action": "searchHotels",
                "inputs": {
                    "area": "Business district",
                    "check-in-date": "2024-11-15",
                    "number-of-nights": 4,
                },
                "enter_guard": "True",
                "success_criteria": "len(result) > 0",
                "depends_on": [1],
                "next_step": 3,
            },
            {
                "step_id": 3,
                "title": "Flight from New York City to Chicago",
                "action": "searchFlights",
                "inputs": {
                    "from_airport_code": "JFK",
                    "to_airport_code": "ORD",
                    "outbound_date": "2024-11-19",
                    "passengers": 3,
                },
                "enter_guard": "True",
                "success_criteria": "len(result) > 0 and result[0].get('flight')",
                "depends_on": [2],
                "next_step": 4,
            },
            {
                "step_id": 4,
                "title": "Transfer from ORD Airport to Hotel in Chicago",
                "action": "searchTransfers",
                "inputs": {"airport": "ORD", "hotel": "Business district Chicago"},
                "enter_guard": "True",
                "success_criteria": "result.get('confirmed') == True",
                "depends_on": [3],
                "next_step": 5,
            },
            {
                "step_id": 5,
                "title": "Stay at Hotel in Chicago",
                "action": "searchHotels",
                "inputs": {
                    "area": "Business district Chicago",
                    "check-in-date": "2024-11-19",
                    "number-of-nights": 3,
                },
                "enter_guard": "True",
                "success_criteria": "len(result) > 0",
                "depends_on": [4],
                "next_step": 6,
            },
            {
                "step_id": 6,
                "title": "Flight from Chicago to Rio de Janeiro",
                "action": "searchFlights",
                "inputs": {
                    "from_airport_code": "ORD",
                    "to_airport_code": "GIG",
                    "outbound_date": "2024-11-22",
                    "passengers": 3,
                },
                "enter_guard": "True",
                "success_criteria": "len(result) > 0 and result[0].get('flight')",
                "depends_on": [5],
                "next_step": None,
            },
        ],
        "meta": {"strategy": "compose"},
    }

    executor = ExecutePlan({})

    summary = executor.run(plan)
    print(json.dumps(summary, indent=2, cls=DecimalEncoder))


