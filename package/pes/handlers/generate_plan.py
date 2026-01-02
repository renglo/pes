from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol
import json, math, time, uuid, re
import os
from openai import OpenAI
from collections import Counter
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
    
    

# ────────────────────────────────────────────────────────────────────────────────
# Core data shapes
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Signature:
    domain: Optional[str] = None
    goal: Optional[str] = None
    destination: Optional[str] = None
    party: Optional[Dict[str, Any]] = None
    dates: Optional[Dict[str, str]] = None
    constraints: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    activities: Optional[List[Dict[str, Any]]] = None  # Requested activities/services
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

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

@dataclass
class Case:
    id: str
    signature_text: str
    plan: Plan
    outcomes: Dict[str, Any]
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    

# ────────────────────────────────────────────────────────────────────────────────
# Embedding + cosine (simple; swap with your prod embedder)
# ────────────────────────────────────────────────────────────────────────────────

class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...

class SimpleEmbedder:
    def __init__(self):
        self.vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        try:
            obj = json.loads(text)
            out = []
            def walk(prefix, v):
                if isinstance(v, dict):
                    for k, vv in v.items(): walk(f"{prefix}{k}.", vv)
                elif isinstance(v, list):
                    for i, vv in enumerate(v): walk(f"{prefix}{i}.", vv)
                else:
                    out.append(f"{prefix[:-1]}:{str(v).lower()}")
            walk("", obj)
            return out
        except Exception:
            return re.findall(r"[a-z0-9_]+", text.lower())

    def embed(self, text: str) -> List[float]:
        toks = self._tokenize(text)
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        vec = [0.0]*len(self.vocab)
        c = Counter(toks)
        for t, n in c.items():
            vec[self.vocab[t]] = float(n)
        s = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x/s for x in vec]

def cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i]*b[i] for i in range(n))
    na = math.sqrt(sum(x*x for x in a[:n])) or 1.0
    nb = math.sqrt(sum(x*x for x in b[:n])) or 1.0
    return dot/(na*nb)
    
    

# ────────────────────────────────────────────────────────────────────────────────
# Vector DB (mock)
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class VDBItem:
    id: str
    kind: str            # "case" | "fact" | "skill"
    text: str            # serialized content (or free text) for LLM context
    meta: Dict[str, Any] # filters: destination/season/etc.
    vec: List[float]     # embedding

class VectorDB:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.items: List[VDBItem] = []

    def add(self, kind: str, text: str, meta: Optional[Dict[str, Any]] = None) -> str:
        vid = f"vdb_{kind}_{uuid.uuid4().hex[:8]}"
        vec = self.embedder.embed(text)
        self.items.append(VDBItem(id=vid, kind=kind, text=text, meta=meta or {}, vec=vec))
        return vid

    def search(self, query: str, kind: Optional[str] = None, k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[VDBItem]:
        q = self.embedder.embed(query)
        scored: List[Tuple[VDBItem, float]] = []
        for it in self.items:
            if kind and it.kind != kind:
                continue
            # For demo: compute similarity score, boost by 0.5 if filters match exactly
            s = cosine(q, it.vec)
            if filters:
                # Boost score if all filters match, but don't exclude non-matching items
                if all(it.meta.get(k1) == v1 for k1, v1 in filters.items() if v1 is not None):
                    s += 0.5  # Boost matching items
            scored.append((it, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored[:k]] 
    
    

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
        # Minimal but strict schemas for the agent's JSON outputs
        if task == "TO_SIGNATURE":
            return {
                "name": "Signature",
                "schema": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "goal": {"type": "string"},
                        "destination": {"type": ["string","null"]},
                        "party": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": False},
                                {"type": "null"}
                            ]
                        },
                        "dates": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": False},
                                {"type": "null"}
                            ]
                        },
                        "constraints": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": False},
                                {"type": "null"}
                            ]
                        },
                        "preferences": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": False},
                                {"type": "null"}
                            ]
                        },
                        "activities": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False
                                    }
                                },
                                {"type": "null"}
                            ]
                        },
                        "extras": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": False},
                                {"type": "null"}
                            ]
                        }
                    },
                    "required": ["domain","goal","destination","party","dates","constraints","preferences","activities","extras"],
                    "additionalProperties": False
                },
                "strict": True
            }
        if task in ("ADAPT_PLAN", "COMPOSE_PLAN"):
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
        if task == "SKILL_SCORING":
            return {
                "name": "SkillScores",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "skill_id": {"type": "string"},
                                    "score": {"type": "number"}
                                },
                                "required": ["skill_id","score"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["scores"],
                    "additionalProperties": False
                },
                "strict": True
            }
        if task == "SELECT_BEST_PLAN":
            return {
                "name": "PlanChoice",
                "schema": {
                    "type": "object",
                    "properties": {
                        "choice_index": {"type": "integer"},
                        "reason": {"type": "string"}
                    },
                    "required": ["choice_index"],
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
        We still nudge the model to return JSON if the task indicates so.
        """
        task = self._detect_task(prompt)
        # If SELECT_BEST_PLAN, ask for structured JSON
        response_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature":0.2
        }
        if task == "SELECT_BEST_PLAN":
            if self._supports_structured_outputs():
                response_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": self._schema_for_task(task)
                }
            else:
                # For older models, use JSON mode and include schema in prompt
                schema = self._schema_for_task(task)
                schema_str = json.dumps(schema["schema"], indent=2)
                enhanced_prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{schema_str}"
                response_kwargs["response_format"] = {"type": "json_object"}
                response_kwargs["messages"][0]["content"] = enhanced_prompt
        
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
# PlanAgent
# ────────────────────────────────────────────────────────────────────────────────

class PlanAgent:
    """
    v2.2:
      - LLM-crafted signature, plan adaptation/composition, and selection
      - VectorDB retrieval for cases, facts, skills
      - Action catalog provided to LLM; validator enforces action/arg correctness
    """
    def __init__(self,
                 vdb: VectorDB,
                 llm: LLMClient,
                 action_catalog: List[ActionSpec],
                 prompts: Optional[Dict[str, str]] = None):
        self.vdb = vdb
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
            'sig_text', 'example_cases', 'fact_texts', 'catalog_summary',
            'catalog', 'skills', 'activities_requested', 'plan_details'
        }
        
        for token, value in replacements.items():
            # Handle #token# format (new, preferred)
            token_placeholder = '#' + token + '#'
            if token_placeholder in prompt:
                if token in json_tokens:
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
                if token in json_tokens:
                    replacement = f'```json\n{value}\n```'
                else:
                    replacement = str(value)
                prompt = prompt.replace(legacy_placeholder1, replacement)
                prompt = prompt.replace(legacy_placeholder2, replacement)
        
        return prompt

    # 1) Signature via LLM
    def to_signature(self, req: Dict[str, Any]) -> Signature:
        print('Turning your request into a structured signature...')
        
        # Extract the actual request text
        request_text = req.get("request", "") if isinstance(req.get("request"), str) else json.dumps(req)
        print(f'[DEBUG] Request text: {request_text}')
        
        # Current time and date
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        print(f'[DEBUG] Current time computed: {current_time}')
        
        # Use prompt from database if available, otherwise use default
        prompt_template = self.prompts.get('to_signature', '')
        if prompt_template:
            print(f'[DEBUG] Using prompt from YAML (length: {len(prompt_template)} chars)')
            # Replace tokens with computed values
            replacements = {
                'request_text': request_text,
                'current_time': current_time
            }
            print(f'[DEBUG] Token replacements: {replacements}')
            prompt = self._replace_tokens(prompt_template, replacements)
            # Verify token replacement
            if '#current_time#' in prompt or '{current_time}' in prompt:
                print(f'[WARNING] Token #current_time# or {{current_time}} still present in prompt after replacement!')
            if '#request_text#' in prompt or '{request_text}' in prompt:
                print(f'[WARNING] Token #request_text# or {{request_text}} still present in prompt after replacement!')
            print(f'[DEBUG] Prompt preview (first 500 chars): {prompt[:500]}...')
        else:
            print('[WARNING] No prompt template found, using fallback hardcoded prompt')
            # Fallback to default hardcoded prompt (for backward compatibility)
            prompt = f"""
            You are a systems normalizer for an experience-retrieval planning agent specializing in complex tasks.
            TASK: TO_SIGNATURE
            Goal: Analyze the user's request and extract structured information into a normalized Signature.
            
            CRITICAL: Read the USER REQUEST carefully and extract ALL details mentioned.
            
            Return ONLY a compact JSON with keys like:
            ["domain","goal","destination","party","dates","constraints","preferences","activities","extras"]
            
            REQUIRED FIELDS (never leave these as null):
            - domain: Always "travel" for travel requests
            - goal: Brief description of trip purpose (e.g., "business meeting", "conference", "client visit")
            - destination: The city/location they're traveling TO
            - party: Object with traveler count and roles (e.g., {{"travelers": 4, "roles": ["executive"]}})
            - activities: Array of requested activities/services with structured details
            - extras: MUST include "origin" (where they're traveling FROM) and "trip_type"
            
            ACTIVITIES FIELD:
            - Extract ALL requested activities, tours, events, or services
            - Structure each activity as: {{"type": "activity_type", "description": "details", "location": "where", "when": "timing"}}
            - Common types: "city_tour", "restaurant", "theater", "museum", "day_trip", "conference", "meeting", "spa", "shopping"
            - Examples:
              * "city tour" → {{"type": "city_tour", "description": "guided city tour", "location": "destination_city"}}
              * "dinner in good restaurant" → {{"type": "restaurant", "description": "dinner at upscale restaurant", "preferences": ["fine_dining"]}}
              * "theater show" → {{"type": "theater", "description": "Broadway show", "location": "destination_city"}}
            - If no activities requested, set to empty array []
            
            HANDLING DIFFERENT TRIP TYPES:
            
            1. SIMPLE SINGLE-CITY TRIP (Origin → Destination → Origin, no other cities visited):
               - destination: "CityName"
               - extras.origin: "OriginCity"
               - extras.trip_type: "single_destination"
               - NO itinerary array needed (simple round trip)
               - activities array contains what they do IN that one city
            
            2. MULTI-CITY TRIP (A → B → C → ... → A):
               - destination: "PrimaryDestination"  (where they're staying longest or main city)
               - extras.origin: "StartCity"
               - extras.trip_type: "multi_city"
               - extras.itinerary: [
                   {{"from": "CityA", "to": "CityB", "date": "YYYY-MM-DD"}},
                   {{"from": "CityB", "to": "CityC", "date": "YYYY-MM-DD"}},
                   {{"from": "CityC", "to": "CityA", "date": "YYYY-MM-DD"}}  // return leg
                 ]
               - activities array contains what they do at each city
               - INCLUDES day trips to other cities during a main stay:
                 * Example: Stay in NYC, day trip to Philadelphia
                 * trip_type: "multi_city" (because visiting multiple cities)
                 * itinerary includes: origin → NYC → Philadelphia → NYC → origin
                 * activities includes the day trip activity
            
            3. CONVERGING TRAVELERS (Multiple origins → Single destination):
               - destination: "MeetingCity"
               - extras.trip_type: "converging"
               - extras.travelers: [
                   {{"count": N, "origin": "CityX", "arrival_date": "YYYY-MM-DD"}},
                   {{"count": M, "origin": "CityY", "arrival_date": "YYYY-MM-DD"}}
                 ]
               - extras.return_date: "YYYY-MM-DD"  (common return or per traveler)
            
            4. DAY TRIP / SAME-DAY RETURN (Origin → Destination → Origin SAME DAY, no overnight):
               - Use ONLY when the ENTIRE TRIP is same-day return from origin
               - destination: "CityName"
               - extras.origin: "OriginCity"
               - extras.trip_type: "day_trip"
               - extras.venue: "VenueName or Address"  (optional - where they're going)
               - dates: {{"date": "YYYY-MM-DD"}}  (same day for both flights)
               - activities array contains what they do during the day trip
               - NOTE: If staying in City A and taking day trip to City B, that's "multi_city", NOT "day_trip"
            
            EXTRACTION RULES:
            - ALWAYS set domain to "travel"
            - Identify trip purpose for goal field
            - Identify if trip involves multiple destinations in sequence (multi-city)
            - Identify if multiple travelers are coming from different origins (converging)
            - Identify if same-day return (day trip)
            - Extract ALL cities mentioned: origin city (where FROM) and destination city (where TO)
            - Extract ALL dates with their associated legs
            - For multi-city: create complete itinerary including return leg to origin
            - Extract party details (total count, roles, special needs)
            - Include any special requirements (wheelchair, dietary, etc.) in constraints or party
            
            CRITICAL: ITINERARY vs ACTIVITIES:
            - extras.itinerary = Physical city-to-city movements (creates flights/transportation)
            - activities = What travelers DO at each location (tours, dinners, meetings)
            - A "day trip to Philadelphia" creates BOTH:
              * extras.itinerary entry: NYC → Philadelphia → NYC (with dates)
              * activities entry: {{"type": "day_trip", "location": "Philadelphia"}}
            - ALWAYS create itinerary entries when request mentions visiting other cities
            
            DATES FORMAT:
            - Convert relative dates to approximate YYYY-MM-DD format
            - For "X days" duration, calculate return date
            - If no specific date mentioned, use "estimated_date"
            - Consider the current time and date when scheduling
            - Current time is : "{current_time}"
            
            USER REQUEST TEXT:
            "{request_text}"
            
            ANALYZE THE REQUEST ABOVE AND EXTRACT:
            1. How many people are traveling?
            2. What are their roles? (executives, technicians, etc.)
            3. Where are they traveling FROM? (origin)
            4. Where are they traveling TO? (primary destination)
            5. Are they visiting OTHER CITIES during the trip? (creates itinerary + activities)
            6. When? (dates)
            7. Is it same-day return from origin, overnight stay, or multi-city journey?
            8. What's the purpose?
            9. What activities/services are requested? (tours, dinners, meetings, etc.)
            
            EXAMPLES:
            
            Request: "Fly from Boston to NYC for 3 days"
            Output: {{
              "domain": "travel",
              "goal": "leisure trip",
              "destination": "New York City",
              "activities": [],
              "extras": {{
                "origin": "Boston",
                "trip_type": "single_destination"
              }},
              "dates": {{"departure": "...", "duration_days": 3}}
            }}
            
            Request: "Group from Milan wants NYC trip with city tour and dinner"
            Output: {{
              "domain": "travel",
              "goal": "group leisure trip",
              "destination": "New York City",
              "activities": [
                {{"type": "city_tour", "description": "guided city tour", "location": "New York City"}},
                {{"type": "restaurant", "description": "dinner at upscale restaurant", "preferences": ["fine_dining"]}}
              ],
              "extras": {{
                "origin": "Milan",
                "trip_type": "single_destination"
              }}
            }}
            
            Request: "42 people from Milan to NYC for 4 nights, want city tour, dinner, and day trip to Philadelphia"
            Output: {{
              "domain": "travel",
              "goal": "group leisure trip",
              "destination": "New York City",
              "party": {{"travelers": 42}},
              "activities": [
                {{"type": "city_tour", "description": "guided city tour", "location": "New York City"}},
                {{"type": "restaurant", "description": "dinner at upscale restaurant", "preferences": ["fine_dining"]}},
                {{"type": "day_trip", "description": "day trip to Philadelphia", "location": "Philadelphia"}}
              ],
              "extras": {{
                "origin": "Milan",
                "trip_type": "multi_city",
                "itinerary": [
                  {{"from": "Milan", "to": "New York City", "date": "arrival_date"}},
                  {{"from": "New York City", "to": "Philadelphia", "date": "during_stay"}},
                  {{"from": "Philadelphia", "to": "New York City", "date": "same_day"}},
                  {{"from": "New York City", "to": "Milan", "date": "departure_date"}}
                ]
              }},
              "dates": {{"start": "arrival_date", "end": "departure_date", "duration_nights": 4}}
            }}
            
            Request: "Two technicians fly from Sao Paulo to Denver (Nov 20) then Newark (Nov 23)"
            Output: {{
              "domain": "travel",
              "destination": "Newark",
              "party": {{"travelers": 2, "roles": ["technician"]}},
              "extras": {{
                "origin": "Sao Paulo",
                "trip_type": "multi_city",
                "itinerary": [
                  {{"from": "Sao Paulo", "to": "Denver", "date": "2026-11-20"}},
                  {{"from": "Denver", "to": "Newark", "date": "2026-11-23"}},
                  {{"from": "Newark", "to": "Sao Paulo", "date": "estimated_return"}}
                ]
              }}
            }}
            
            Request: "3 people from Boston and 2 from Chicago meeting in NYC for conference Dec 5-8"
            Output: {{
              "domain": "travel",
              "destination": "New York City",
              "party": {{"total_travelers": 5}},
              "extras": {{
                "trip_type": "converging",
                "travelers": [
                  {{"count": 3, "origin": "Boston", "arrival_date": "2026-12-05"}},
                  {{"count": 2, "origin": "Chicago", "arrival_date": "2026-12-05"}}
                ],
                "return_date": "2026-12-08"
              }},
              "dates": {{"start": "2026-12-05", "end": "2026-12-08"}}
            }}
            
            Request: "Four executives fly to Lima from Sao Paulo in the morning and back the same day"
            Output: {{
              "domain": "travel",
              "destination": "Lima",
              "party": {{"travelers": 4, "roles": ["executive"]}},
              "extras": {{
                "origin": "Sao Paulo",
                "trip_type": "day_trip"
              }},
              "dates": {{"date": "estimated_date"}}
            }}
        """
        # Note: Token replacement is already handled by _replace_tokens() above
        # This fallback section uses hardcoded prompt with old format for backward compatibility
        data = self.llm.complete_json(prompt)
        
        if not data:
            print('[ERROR] LLM returned no data for signature')
            return False
        
        print('[DEBUG] LLM response for signature:', json.dumps(data, indent=2))
        print('Parsed signature:', json.dumps(data, indent=2))
        
        # Filter to only valid Signature fields
        valid_fields = {'domain', 'goal', 'destination', 'party', 'dates', 'constraints', 'preferences', 'activities', 'extras'}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return Signature(**filtered_data)

    # 2) Retrieval via VectorDB
    def retrieve(self, sig: Signature, k_cases: int = 4, k_facts: int = 4, k_skills: int = 6):
        print('Retrieving cases, fact and skills from agent experience...')
        query = sig.to_text()
        filters = {"destination": (sig.destination or None)}
        filt = {k:v for k, v in filters.items() if v}
        cases = self.vdb.search(query=query, kind="case", k=k_cases, filters=filt)
        facts = self.vdb.search(query=query, kind="fact", k=k_facts, filters=filt)
        skills = self.vdb.search(query=query, kind="skill", k=k_skills, filters=filt)

        # LLM skill scoring
        prompt_scores = f"""
            You are a skill ranker.
            TASK: SKILL_SCORING
            Given a Signature and a list of candidate skills, score each skill's applicability in [0,1].
            Return JSON: {{"scores": [{{"skill_id": str, "score": float}}]}}

            Signature:
            ```json
            {sig.to_text()}
            ```

            Skills:
            ```json
            {json.dumps([{"id": s.id, "text": s.text, "meta": s.meta} for s in skills], indent=2)}
            ```
        """
        scored = self.llm.complete_json(prompt_scores).get("scores", [])
        #if not scored:
        #    return False
        
        score_map = {s["skill_id"]: s["score"] for s in scored}
        skills_ranked = sorted(skills, key=lambda s: score_map.get(s.id, 0), reverse=True)
        #print('Cases:',cases)
        #print('Facts:',facts)
        #print('Skills:',skills_ranked)
        
        return {'cases':cases, 'facts':facts, 'skills':skills_ranked}

    # 3) Adapt plan via LLM using multiple example cases, then validate
    def adapt_from_cases(self, cases: List[Case], sig: Signature, facts: List[VDBItem]) -> Plan:
        print(f'Adapting plan from {len(cases)} example cases...')
        
        # Convert cases to example format
        example_cases = []
        for case in cases[:3]:  # Use up to 3 example cases
            example_cases.append({
                "signature": case.signature_text,
                "plan_steps": [asdict(s) for s in case.plan.steps]
            })
        
        #print('Example cases:', json.dumps(example_cases, indent=2))
        fact_texts = [{"text": f.text, "meta": f.meta} for f in facts]
        
        # Build action catalog summary for reference
        catalog_summary = [{"name": t.key, "required_args": t.required_args} for t in self.action_catalog]
        
        # Use prompt from database if available, otherwise use default
        prompt_template = self.prompts.get('adapt_plan', '')
        if prompt_template:
            # Compute values for tokens
            sig_text = sig.to_text()
            example_cases_json = json.dumps(example_cases, indent=2)
            fact_texts_json = json.dumps(fact_texts, indent=2)
            catalog_summary_json = json.dumps(catalog_summary, indent=2)
            plan_id = uuid.uuid4().hex[:8]
            
            # Replace tokens with computed values
            replacements = {
                'sig_text': sig_text,
                'example_cases': example_cases_json,
                'fact_texts': fact_texts_json,
                'catalog_summary': catalog_summary_json,
                'plan_id': plan_id
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            # Fallback to default hardcoded prompt
            prompt = f"""
            You are an expert travel planner using case-based reasoning.
            TASK: ADAPT_PLAN
            
            INSTRUCTIONS:
            1. Study the EXAMPLE CASES below to understand travel planning patterns and structure
            2. Analyze the NEW REQUEST to determine trip type (single-city, multi-city, or converging travelers)
            3. Generate a COMPLETE NEW plan using patterns but with CORRECT details from the new request
            
            CRITICAL COMPLETENESS RULE:
            - EVERY plan MUST include a return flight to the ORIGINAL ORIGIN (where the journey started)
            - If there are intermediate activities (tours, day trips, side trips), include them BUT also the final return home
            - Example: Milan → NYC (stay 4 nights + activities) → Milan (return flight)
            - Example: Milan → NYC → Philadelphia (day trip) → NYC → Milan (return flight)
            
            TRIP TYPE HANDLING:
            
            A. SINGLE-CITY TRIP (extras.trip_type == "single_destination"):
               - Outbound flight: origin → destination
               - Hotel at destination
               - [Include REQUESTED ACTIVITIES from signature.activities array - see below]
               - Return flight: destination → origin (MUST ALWAYS INCLUDE)
            
            INCORPORATING ACTIVITIES:
            - Check signature.activities array for requested services
            - For each activity in the array, create appropriate plan steps
            - Examples:
              * {{"type": "city_tour"}} → Add step using appropriate tour action
              * {{"type": "restaurant"}} → Add step for restaurant reservation
              * {{"type": "day_trip", "location": "OtherCity"}} → Add round-trip flights + activity
            - Place activities in chronological order during the stay
            - If actions don't exist for an activity, note it in plan.meta but don't skip it
            
            B. MULTI-CITY TRIP (extras.trip_type == "multi_city"):
               - Read extras.itinerary array for complete journey
               - For each leg in itinerary:
                 * Flight from → to with specified date
                 * Hotel booking (if overnight stay)
               - CRITICAL: Follow the EXACT sequence in itinerary including return leg
            
            C. CONVERGING TRAVELERS (extras.trip_type == "converging"):
               - Read extras.travelers array
               - For EACH traveler group:
                 * Flight from their origin → destination
               - Single hotel booking at destination
               - For EACH traveler group:
                 * Return flight destination → their origin
            
            D. DAY TRIP (extras.trip_type == "day_trip"):
               - Outbound flight: origin → destination (morning/early)
               - NO HOTEL (same day return)
               - Return flight: destination → origin (evening/late)
            
            E. SIDE TRIP DURING MAIN TRIP:
               - If request mentions activities/day trips to OTHER CITIES during a main stay:
                 * Treat as intermediate activities within the main trip
                 * Include round-trip to the side destination (e.g., NYC → Philadelphia → NYC)
                 * Then continue with main trip completion (NYC → origin)
                 * Example: Milan → NYC [stay] → Philadelphia [day trip] → NYC [continue stay] → Milan [return]
            
            AIRPORT CODES:
            - Determine correct airport codes for ALL cities (e.g., GRU for Sao Paulo, DEN for Denver, EWR for Newark)
            - Use major airports for each city
            
            DATES & PASSENGERS:
            - Use dates from signature (dates field or extras.itinerary)
            - Use passenger count from party field
            - For multi-city: use date from each itinerary leg
            - For converging: use arrival_date for each traveler group
            
            FLIGHT LEG PARAMETER:
            - For ALL quote_flight actions, include the "leg" parameter in inputs
            - The leg represents the chronological order of flights in the trip (0-based index)
            - First flight chronologically = leg: 0, second flight = leg: 1, third = leg: 2, etc.
            - For single-city trips: outbound flight = leg: 0, return flight = leg: 1
            - For multi-city trips: assign leg numbers in chronological order based on departure dates
            - For day trips: outbound = leg: 0, return = leg: 1
            - For converging travelers: assign leg numbers chronologically across all traveler groups
            - Example: Sao Paulo → Denver (leg: 0), Denver → Newark (leg: 1), Newark → Sao Paulo (leg: 2)
            
            NEW REQUEST (use these details for all inputs):
            ```json
            {sig.to_text()}
            ```
            
            EXAMPLE CASES (learn patterns and structure, NOT specific values):
            ```json
            {json.dumps(example_cases, indent=2)}
            ```
            
            RELEVANT FACTS:
            ```json
            {json.dumps(fact_texts, indent=2)}
            ```
            
            ACTION CATALOG (ensure inputs match required_args):
            ```json
            {json.dumps(catalog_summary, indent=2)}
            ```
            
            OUTPUT FORMAT:
            Return ONLY JSON with this exact structure:
            {{
                "plan": {{
                    "id": "{uuid.uuid4().hex[:8]}",
                    "meta": {{"strategy": "adapted"}},
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
                        }},
                        {{
                            "step_id": 1,
                            "title": "next step title",
                            "action": "actionName",
                            "inputs": {{"arg1": "value1", ...}},
                            "enter_guard": "True",
                            "success_criteria": "appropriate criteria",
                            "depends_on": [0],
                            "next_step": null
                        }}
                    ]
                }}
            }}
            
            STEP STRUCTURE REQUIREMENTS:
            - step_id: Sequential integer starting from 0 (0, 1, 2, 3, ...)
            - depends_on: Array of step_id integers that this step depends on
            - next_step: The step_id of the next step in execution order, or null for the last step
            
            STEP TITLE FORMATTING (CRITICAL):
            - Use directional titles that clearly indicate FROM → TO cities/airports
            - Format: "[City/Airport] to [City/Airport] flight" (e.g., "Seattle to Orlando flight")
            - NEVER use the word "return" in step titles - flights are booked independently without trip context
            - Examples:
              * Outbound: "Seattle to Orlando flight" (NOT "Return flight from Seattle to Orlando")
              * Intermediate: "Orlando to Miami flight" (NOT "Return flight from Orlando to Miami")
              * Final leg: "Miami to Seattle flight" (NOT "Miami to Seattle return flight")
            - For hotels: Use "[City] hotel booking" or "[Area] hotel"
            - For activities: Use descriptive names like "City tour in [Location]" or "Restaurant reservation"
            - Titles must be unambiguous - each step executes independently without full trip context
            
            CHRONOLOGICAL ORDER (CRITICAL):
            - Steps MUST be listed in the order they occur in time
            - step_id values MUST be sequential: 0, 1, 2, 3...
            - next_step links to the following step (step 0 → 1 → 2 → 3 → null)
            - Correct sequence: Arrival flight → Hotel → Departure flight
            
            
            CRITICAL: Check extras.trip_type and extras.itinerary/travelers to determine correct plan structure!
            
            FINAL VALIDATION BEFORE RETURNING:
            - Does the LAST step return travelers to their ORIGINAL ORIGIN?
            - If origin is "Milan" and destination is "NYC", the final flight MUST be NYC → Milan
            - If there are side trips (e.g., Philadelphia), those are in the middle, not the end
            - The plan should end where it started!
        """
        data = self.llm.complete_json(prompt)
        if not data or "plan" not in data:
            print('[ERROR] LLM returned invalid plan structure')
            return Plan(id=uuid.uuid4().hex[:8], steps=[], meta={"strategy": "adapted", "error": "LLM returned invalid plan"})
        
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
        plan = Plan(id=data["plan"]["id"], steps=steps, meta=data["plan"].get("meta", {}))
        print('Raw adapted plan:')
        print(json.dumps({"id": plan.id, "meta": plan.meta, "steps": [asdict(s) for s in plan.steps]}, indent=2, cls=DecimalEncoder))
        validated_plan = self._validate_and_patch_plan(plan)
        #print('Validated adapted plan:')
        #print(json.dumps({"id": validated_plan.id, "meta": validated_plan.meta, "steps": [asdict(s) for s in validated_plan.steps]}, indent=2, cls=DecimalEncoder))
        return validated_plan

    # 4) Compose plan via LLM with explicit Action Catalog, then validate
    def compose_from_skills(self, sig: Signature, skills: List[VDBItem]) -> Plan:
        print('Composing a new plan from scratch based on the user request...')
        catalog = [{
            "name": t.key,
            "description": t.description,
            "required_args": t.required_args,
            "optional_args": t.optional_args,
            "success_criteria_hint": t.success_criteria_hint
        } for t in self.action_catalog]

        # Use prompt from database if available, otherwise use default
        prompt_template = self.prompts.get('compose_plan', '')
        if prompt_template:
            # Compute values for tokens
            sig_text = sig.to_text()
            catalog_json = json.dumps(catalog, indent=2)
            skills_json = json.dumps([{"id": s.id, "text": s.text, "meta": s.meta} for s in skills], indent=2)
            plan_id = uuid.uuid4().hex[:8]
            
            # Replace tokens with computed values
            replacements = {
                'sig_text': sig_text,
                'catalog': catalog_json,
                'skills': skills_json,
                'plan_id': plan_id
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            # Fallback to default hardcoded prompt
            prompt = f"""
            You are a planner that composes a COMPLETE executable travel plan using ONLY the allowed actions.
            TASK: COMPOSE_PLAN
            
            CRITICAL COMPLETENESS RULE:
            - EVERY plan MUST include a return flight to the ORIGINAL ORIGIN (where the journey started)
            - If there are intermediate activities (tours, day trips, side trips), include them BUT also the final return home
            - The plan is NOT complete until travelers return to where they started
            - Example: Milan → NYC (stay 4 nights + city tour + day trip to Philadelphia) → Milan (return flight)
            
            ANALYZE TRIP TYPE from Signature extras.trip_type:
            
            A. SINGLE-CITY TRIP (extras.trip_type == "single_destination"):
               Required sequence:
               1. Flight: origin → destination
               2. Hotel: at destination
               3. [ACTIVITIES - read from signature.activities array and create steps for each]
               4. Flight: destination → origin (MUST ALWAYS INCLUDE - return to starting point)
            
            INCORPORATING ACTIVITIES:
            - Check signature.activities array for requested services
            - For each activity, create steps using available actions or note in plan.meta
            - Common activity types and how to handle them:
              * "city_tour" → Use tour action if available, or note as manual booking needed
              * "restaurant" → Use restaurant booking action if available, or note as manual reservation
              * "day_trip" with location → Create round-trip flights for that day
              * "theater", "museum", "spa" → Note as manual booking needed (no actions yet)
            - Place activity steps in chronological order during the stay
            
            B. MULTI-CITY TRIP (extras.trip_type == "multi_city"):
               - Read extras.itinerary array
               - For EACH leg in itinerary:
                 1. Flight: from → to (with date from itinerary)
                 2. Hotel: at city (if overnight)
               - MUST cover ALL legs in itinerary including return
               - Example: Sao Paulo → Denver → Newark → Sao Paulo = 3 flights
            
            C. CONVERGING TRAVELERS (extras.trip_type == "converging"):
               - Read extras.travelers array
               - Create separate flight  for EACH traveler group origin
               - Single shared hotel at destination
            
            D. DAY TRIP (extras.trip_type == "day_trip"):
               Required sequence (NO HOTEL):
               1. Flight: origin → destination (morning/early flight)
               2. Flight: destination → origin (evening/return flight)
               - Use same date for both flights
            
            E. SIDE TRIP DURING MAIN TRIP:
               - If request mentions activities/day trips to OTHER CITIES during a main stay:
                 * Treat as intermediate activities within the main trip
                 * Include round-trip to the side destination (e.g., NYC → Philadelphia → NYC)
                 * Then continue with main trip completion (NYC → origin)
                 * Example: Milan → NYC [stay] → Philadelphia [day trip] → NYC → Milan [return]
                 * CRITICAL: Must still include final return to ORIGINAL ORIGIN
            
            AIRPORT DETERMINATION:
            - Find major airport codes for ALL cities mentioned
            - Examples: Sao Paulo=GRU, Denver=DEN, Newark=EWR, NYC=JFK, Chicago=ORD
            
            PASSENGERS:
            - Single/Multi-city: use party.travelers or party.total_travelers
            - Converging: use "count" from each traveler group in extras.travelers
            
            DATES:
            - Single-city: use dates.departure and dates.return
            - Multi-city: use "date" from each leg in extras.itinerary
            - Converging: use arrival_date from extras.travelers, return_date from extras
            
            FLIGHT LEG PARAMETER:
            - For ALL quote_flight actions, include the "leg" parameter in inputs
            - The leg represents the chronological order of flights in the trip (0-based index)
            - First flight chronologically = leg: 0, second flight = leg: 1, third = leg: 2, etc.
            - For single-city trips: outbound flight = leg: 0, return flight = leg: 1
            - For multi-city trips: assign leg numbers in chronological order based on departure dates
            - For day trips: outbound = leg: 0, return = leg: 1
            - For converging travelers: assign leg numbers chronologically across all traveler groups
            - Example: Sao Paulo → Denver (leg: 0), Denver → Newark (leg: 1), Newark → Sao Paulo (leg: 2)
            
            Signature (CHECK extras.trip_type and extras.itinerary/travelers!):
            ```json
            {sig.to_text()}
            ```

            ACTION_CATALOG:
            ```json
            {json.dumps(catalog, indent=2)}
            ```

            Top Skills (semantic hints):
            ```json
            {json.dumps([{"id": s.id, "text": s.text, "meta": s.meta} for s in skills], indent=2)}
            ```

            Return ONLY JSON:
            {{"plan": {{"id":"{uuid.uuid4().hex[:8]}", "meta": {{"strategy":"compose"}}, "steps": [PlanStep..]}}}}
            
            Each PlanStep:
            {{
            "step_id": 0,
            "title": "descriptive title",
            "action": "<actionName from ACTION_CATALOG>",
            "inputs": {{"required_arg": "value", ...}},
            "enter_guard": "True",
            "success_criteria": "from action hint",
            "depends_on": [],
            "next_step": 1
            }}
            
            STEP STRUCTURE REQUIREMENTS:
            - step_id: Sequential integer starting from 0 (0, 1, 2, 3, ...)
            - depends_on: Array of step_id integers that this step depends on
            - next_step: The step_id of the next step in execution order, or null for the last step
            
            STEP TITLE FORMATTING (CRITICAL):
            - Use directional titles that clearly indicate FROM → TO cities/airports
            - Format: "[City/Airport] to [City/Airport] flight" (e.g., "Seattle to Orlando flight")
            - NEVER use the word "return" in step titles - flights are booked independently without trip context
            - Examples:
              * Outbound: "Seattle to Orlando flight" (NOT "Return flight from Seattle to Orlando")
              * Intermediate: "Orlando to Miami flight" (NOT "Return flight from Orlando to Miami")
              * Final leg: "Miami to Seattle flight" (NOT "Miami to Seattle return flight")
            - For hotels: Use "[City] hotel booking" or "[Area] hotel"
            - For activities: Use descriptive names like "City tour in [Location]" or "Restaurant reservation"
            - Titles must be unambiguous - each step executes independently without full trip context
            
            CHRONOLOGICAL ORDER (CRITICAL):
            - Steps MUST be listed in the order they occur in time
            - step_id values MUST be sequential: 0, 1, 2, 3...
            - next_step links to the following step (step 0 → 1 → 2 → 3 → null)
            - Correct sequence: Arrival flight → Hotel → Departure flight
            
            
            CRITICAL: For multi-city trips, include a flight for EVERY leg in extras.itinerary!
            
            FINAL VALIDATION BEFORE RETURNING:
            - Does the LAST step return travelers to their ORIGINAL ORIGIN?
            - Check extras.origin field to see where the journey started
            - The final flight MUST go back to that origin city
            - If there are side trips or day trips to other cities, those are in the middle, not the end
            - The plan should form a complete round trip!
        """
        data = self.llm.complete_json(prompt)
        if not data or "plan" not in data:
            print('[ERROR] LLM returned invalid plan structure')
            return Plan(id=uuid.uuid4().hex[:8], steps=[], meta={"strategy": "compose", "error": "LLM returned invalid plan"})
        
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
        plan = Plan(id=data["plan"]["id"], steps=steps, meta=data["plan"].get("meta", {}))
        print('Raw composed plan:')
        print(json.dumps({"id": plan.id, "meta": plan.meta, "steps": [asdict(s) for s in plan.steps]}, indent=2, cls=DecimalEncoder))
        validated_plan = self._validate_and_patch_plan(plan)
        #print('Validated composed plan:')
        #print(json.dumps({"id": validated_plan.id, "meta": validated_plan.meta, "steps": [asdict(s) for s in validated_plan.steps]}, indent=2, cls=DecimalEncoder))
        return validated_plan

    # 5) Select best plan via LLM
    def select_plan(self, sig: Signature, candidates: List[Plan]) -> Plan:
        print('Selecting best plan from list of candidates...')
        
        # Create detailed plan summaries including step titles and actions
        plan_details = []
        for i, p in enumerate(candidates):
            step_summaries = []
            for step in p.steps:
                step_summaries.append({
                    "title": step.title,
                    "action": step.action,
                    "step_id": step.step_id
                })
            plan_details.append({
                "index": i,
                "id": p.id,
                "strategy": p.meta.get("strategy"),
                "total_steps": len(p.steps),
                "steps": step_summaries
            })
        
        # Extract activities from signature for comparison
        activities_requested = []
        if sig.activities:
            for act in sig.activities:
                activities_requested.append({
                    "type": act.get("type"),
                    "description": act.get("description"),
                    "location": act.get("location")
                })
        
        # Use prompt from database if available, otherwise use default
        prompt_template = self.prompts.get('select_best_plan', '')
        if prompt_template:
            # Compute values for tokens
            sig_text = sig.to_text()
            activities_json = json.dumps(activities_requested, indent=2)
            plan_details_json = json.dumps(plan_details, indent=2)
            
            # Replace tokens with computed values
            replacements = {
                'sig_text': sig_text,
                'activities_requested': activities_json,
                'plan_details': plan_details_json
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            # Fallback to default hardcoded prompt
            prompt = f"""
            You are a meta-planner evaluating travel plans for completeness and quality.
            TASK: SELECT_BEST_PLAN
            
            Given a Signature with requested activities and multiple candidate plans, choose the best one.
            
            EVALUATION CRITERIA (in order of importance):
            1. ACTIVITY COVERAGE: Does the plan include steps for ALL requested activities?
               - Check signature.activities array
               - Each activity should have corresponding plan steps
               - Missing activities = incomplete plan
            
            2. COMPLETENESS: Does the plan form a complete round trip?
               - Must return to original origin
               - Day trips need round-trip flights (outbound + return)
               - All legs in itinerary covered
            
            3. LOGICAL SEQUENCE: Are steps in correct chronological order?
               - Arrival → Hotel → Activities → Return Flight
            
            4. EFFICIENCY: Reasonable number of steps (not too sparse, not redundant)
            
            Return ONLY JSON: {{"choice_index": int, "reason": str}}
            
            The reason should explicitly state which activities are covered/missing in the chosen plan.

            Signature (with requested activities):
            ```json
            {sig.to_text()}
            ```
            
            Activities Requested:
            ```json
            {json.dumps(activities_requested, indent=2)}
            ```

            Candidate Plans (with step details):
            ```json
            {json.dumps(plan_details, indent=2)}
            ```
            
            IMPORTANT: 
            - A day trip to another city needs BOTH outbound and return flights
            - Tours, restaurants, and other activities need dedicated steps
            - Prefer the plan that covers ALL activities over one that skips some
        """
        out = self.llm.complete(prompt)
        print('Candidate selection results:',out)
        try:
            data = json.loads(out)
            idx = max(0, min(len(candidates)-1, int(data.get("choice_index", 0))))
        except Exception:
            idx = 0
        return candidates[idx]

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



    # Orchestrator
    def propose(self, req: Dict[str, Any]) -> Dict[str, Any]:
        function = 'propose'
        print('Initiating planning process...')
        print(req)
        sig = self.to_signature(req)
        if not sig:
            return {'success':False,'input':req,'output':'Signature could not be generated'}
        
        retrieved= self.retrieve(sig)
        
        if not retrieved:
            return{'success':False,'function':function, 'input':req, 'output':'Could not retrieve'}
        
        cases = retrieved['cases']
        facts = retrieved['facts']
        skills = retrieved['skills']

        candidate_plans: List[Plan] = []
        if cases:
            # Convert VDB cases to Case objects for adaptation
            case_objects = [self._vdb_case_to_case(c) for c in cases[:3]]
            adapted = self.adapt_from_cases(case_objects, sig, facts)
            candidate_plans.append(adapted)
        composed = self.compose_from_skills(sig, skills[:4])
        candidate_plans.append(composed)

        final = self.select_plan(sig, candidate_plans)


        return {
            "function":"propose",
            "success": True,
            "input": req,
            "output": {
                "signature":asdict(sig),
                "plan":asdict(final)    
            }    
        }

    # Helper: convert VDB 'case' item to Case object
    def _vdb_case_to_case(self, item: VDBItem) -> Case:
        try:
            obj = json.loads(item.text)
            steps = [PlanStep(**s) for s in obj["plan"]["steps"]]
            p = Plan(id=f"plan.from_case.{item.id}", steps=steps, meta={"strategy":"retrieved"})
            return Case(id=item.id, signature_text=obj.get("signature",""), plan=p, outcomes={"rating":4.2}, context=item.meta)
        except Exception:
            return Case(id=item.id, signature_text="", plan=Plan(id=f"plan.empty.{item.id}", steps=[], meta={}), outcomes={}, context=item.meta)



# ────────────────────────────────────────────────────────────────────────────────
# Handler implementation
# ────────────────────────────────────────────────────────────────────────────────


# Create a context variable to store the request context
request_context: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())

'''
This class searches for Hotels using the Serp API
'''

class GeneratePlan:
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
    
    def _load_prompts(self, portfolio: str, org: str, prompt_ring: str = "pes_prompts") -> Dict[str, str]:
        """
        Load prompts from database.
        Returns a dictionary with keys: 'to_signature', 'adapt_plan', 'compose_plan', 'select_best_plan'
        """
        prompts = {
            'to_signature': '',
            'adapt_plan': '',
            'compose_plan': '',
            'select_best_plan': ''
        }
        
        try:
            # Get all prompt records from the ring
            response = self.DAC.get_a_b(portfolio, org, prompt_ring, limit=1000)
            if response and 'items' in response:
                for item in response['items']:
                    key = item.get('key', '').lower()
                    # Try different possible field names for prompt text
                    prompt_text = item.get('prompt', '') or item.get('text', '') or item.get('content', '')
                    
                    # Map keys to prompt types
                    if 'to_signature' in key or 'signature' in key:
                        prompts['to_signature'] = prompt_text
                    elif 'adapt_plan' in key or 'adapt' in key:
                        prompts['adapt_plan'] = prompt_text
                    elif 'compose_plan' in key or 'compose' in key:
                        prompts['compose_plan'] = prompt_text
                    elif 'select_best_plan' in key or 'select' in key:
                        prompts['select_best_plan'] = prompt_text
        except Exception as e:
            print(f'Warning: Could not load prompts from database: {str(e)}')
            import traceback
            traceback.print_exc()
        
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
            import traceback
            traceback.print_exc()
        
        return actions
    
    def _load_seed_cases(self, portfolio: str, org: str, case_ring: str = "pes_cases") -> List[Dict[str, Any]]:
        """
        Load seed cases from database (trips/cases ring).
        Returns a list of case dictionaries with 'signature' and 'plan' keys.
        """
        cases = []
        
        try:
            # Get all case records from the ring
            response = self.DAC.get_a_b(portfolio, org, case_ring, limit=1000)
            if response and 'items' in response:
                for item in response['items']:
                    # Expect case records to have 'signature' and 'plan' fields
                    signature_text = item.get('signature', '')
                    plan_data = item.get('plan', {})
                    
                    if signature_text and plan_data:
                        cases.append({
                            'signature': signature_text,
                            'plan': plan_data
                        })
        except Exception as e:
            print(f'Warning: Could not load seed cases from database: {str(e)}')
        
        return cases
            
    
    
    
    def build_plan_generator(self, portfolio: str, org: str, 
                         prompt_ring: str = "pes_prompts",
                         action_ring: str = "schd_actions",
                         case_ring: str = "pes_cases") -> PlanAgent:
        embedder = SimpleEmbedder()
        vdb = VectorDB(embedder)
        # Pass the AgentUtilities instance to AIResponsesLLM
        llm = AIResponsesLLM(self.AGU)

        # Use prompts from initialization if available, otherwise load from database
        if self.prompts:
            prompts = self.prompts
        else:
            prompts = self._load_prompts(portfolio, org, prompt_ring)
        
        # Load actions from database
        action_catalog = self._load_actions(portfolio, org, action_ring)
        
        # Fallback to hardcoded actions if database is empty
        if not action_catalog:
            print('Warning: No actions loaded from database, using fallback actions')
            action_catalog = [
                ActionSpec(
                    key="quote_flight",
                    description="Search and quote flights between two airports for given date and number of passengers.",
                    required_args=["from_airport_code", "to_airport_code", "departure_date"],
                    optional_args=["return_date", "passengers", "cabin_class", "direct_only", "leg"],
                    success_criteria_hint="len(result) > 0 and result[0].get('flight')"
                ),
                ActionSpec(
                    key="quote_hotel",
                    description="Search business hotels in a target area and budget.",
                    required_args=["city", "area", "check_in_date", "number_of_nights"],
                    optional_args=["budget", "amenities"],
                    success_criteria_hint="len(result) > 0"
                ),
                ActionSpec(
                    key="rent_bike",
                    description="Reserve bikes for business travelers or leisure.",
                    required_args=["rider_count", "pickup_area"],
                    optional_args=["bike_type"],
                    success_criteria_hint="result.get('confirmed') == True"
                ),
                ActionSpec(
                    key="map_route",
                    description="Plan a walking/biking route for efficient navigation.",
                    required_args=[],
                    optional_args=["max_minutes", "avoid_hills"],
                    success_criteria_hint=""
                ),
            ]

        # Load seed cases from database
        seed_cases = self._load_seed_cases(portfolio, org, case_ring)
        
        # Add seed cases to VectorDB
        for case_data in seed_cases:
            signature_text = case_data.get('signature', '')
            plan_data = case_data.get('plan', {})
            meta = case_data.get('meta', {})
            
            if signature_text and plan_data:
                vdb.add(kind="case",
                       text=json.dumps({"signature": signature_text, "plan": plan_data}),
                       meta=meta)
        
        # Fallback to hardcoded seed cases if database is empty
        if not seed_cases:
            print('Warning: No seed cases loaded from database, using fallback cases')
            # Seed Case 1: NYC business conference trip (summer, single destination)
            case1_sig = Signature(domain="travel", goal="business conference and meetings", destination="NYC",
                                party={"travelers":2},
                                constraints={"budget":"mid"},
                                preferences={"central_location":True,"wifi":True},
                                activities=[
                                    {"type": "conference", "description": "business conference attendance", "location": "NYC"},
                                    {"type": "leisure", "description": "bike rental for recreation", "location": "Central Park"}
                                ],
                                extras={"season":"summer", "origin":"Boston", "purpose":"conference"})
            case1_steps = [
                asdict(PlanStep(step_id=0, title="Boston to NYC flight",
                                action="quote_flight",
                                inputs={"from_airport_code":"BOS","to_airport_code":"JFK","departure_date":"2026-07-15","passengers":2},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=1)),
                asdict(PlanStep(step_id=1, title="Find Midtown business hotel",
                                action="quote_hotel",
                                inputs={"city":"New York City","area":"Midtown Manhattan","budget":"mid","check_in_date":"2026-07-15","number_of_nights":5},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=2)),
                asdict(PlanStep(step_id=2, title="Reserve bikes for leisure",
                                action="rent_bike",
                                inputs={"rider_count":2,"pickup_area":"Central Park"},
                                enter_guard="True",
                                success_criteria="result.get('confirmed') == True",
                                depends_on=[],
                                next_step=3)),
                asdict(PlanStep(step_id=3, title="NYC to Boston flight",
                                action="quote_flight",
                                inputs={"from_airport_code":"JFK","to_airport_code":"BOS","departure_date":"2026-07-20","passengers":2},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=None)),
            ]
            vdb.add(kind="case",
                    text=json.dumps({"signature": case1_sig.to_text(), "plan": {"steps": case1_steps}}),
                    meta={"destination": "NYC"})

            # Seed Case 2: Multi-city business trip (NYC + Washington DC, winter)
            case2_sig = Signature(domain="travel", goal="multi-city business meetings", destination="NYC,Washington DC",
                                party={"travelers":1},
                                constraints={"budget":"high"},
                                preferences={"business_center":True,"airport_proximity":True},
                                activities=[
                                    {"type": "meeting", "description": "client meetings", "location": "NYC"},
                                    {"type": "meeting", "description": "client meetings", "location": "Washington DC"}
                                ],
                                extras={"season":"winter", "origin":"Chicago", "purpose":"client_meetings"})
            case2_steps = [
                asdict(PlanStep(step_id=0, title="Flight from Chicago to NYC",
                                action="quote_flight",
                                inputs={"from_airport_code":"ORD","to_airport_code":"JFK","departure_date":"2026-12-20","passengers":1},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=1)),
                asdict(PlanStep(step_id=1, title="Book NYC hotel",
                                action="quote_hotel",
                                inputs={"city":"New York City","area":"Midtown Manhattan","budget":"high","check_in_date":"2026-12-20","number_of_nights":3},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=2)),
                asdict(PlanStep(step_id=2, title="Flight from NYC to Washington DC",
                                action="quote_flight",
                                inputs={"from_airport_code":"JFK","to_airport_code":"DCA","departure_date":"2026-12-23","passengers":1},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=3)),
                asdict(PlanStep(step_id=3, title="Book DC hotel",
                                action="quote_hotel",
                                inputs={"city":"Washington DC","area":"Downtown DC","budget":"high","check_in_date":"2026-12-23","number_of_nights":4},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=4)),
                asdict(PlanStep(step_id=4, title="Flight from Washington DC to Chicago",
                                action="quote_flight",
                                inputs={"from_airport_code":"DCA","to_airport_code":"ORD","departure_date":"2026-12-27","passengers":1},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=None)),
            ]
            vdb.add(kind="case",
                    text=json.dumps({"signature": case2_sig.to_text(), "plan": {"steps": case2_steps}}),
                    meta={"destination": "NYC,Washington DC"})

            # Seed Case 3: San Francisco tech conference (spring)
            case3_sig = Signature(domain="travel", goal="tech conference and networking", destination="San Francisco",
                                party={"travelers":1},
                                constraints={"budget":"mid"},
                                preferences={"tech_hub_location":True,"public_transit":True},
                                activities=[
                                    {"type": "conference", "description": "tech conference attendance", "location": "SOMA District"},
                                    {"type": "leisure", "description": "bike rental for transportation", "location": "SOMA"},
                                    {"type": "route_planning", "description": "efficient route to venue", "preferences": ["avoid_hills"]}
                                ],
                                extras={"season":"spring", "origin":"Seattle", "purpose":"conference"})
            case3_steps = [
                asdict(PlanStep(step_id=0, title="Flight from Seattle to SF",
                                action="quote_flight",
                                inputs={"from_airport_code":"SEA","to_airport_code":"SFO","departure_date":"2026-04-10","passengers":1},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=1)),
                asdict(PlanStep(step_id=1, title="Book SOMA district hotel",
                                action="quote_hotel",
                                inputs={"city":"San Francisco","area":"SOMA District","budget":"mid","check_in_date":"2026-04-10","number_of_nights":4},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=2)),
                asdict(PlanStep(step_id=2, title="Plan efficient route to venue",
                                action="map_route",
                                inputs={"max_minutes":30,"avoid_hills":True},
                                enter_guard="True",
                                success_criteria="",
                                depends_on=[],
                                next_step=3)),
                asdict(PlanStep(step_id=3, title="Reserve bike for transportation",
                                action="rent_bike",
                                inputs={"rider_count":1,"pickup_area":"SOMA"},
                                enter_guard="True",
                                success_criteria="result.get('confirmed') == True",
                                depends_on=[],
                                next_step=4)),
                asdict(PlanStep(step_id=4, title="Flight from SF to Seattle",
                                action="quote_flight",
                                inputs={"from_airport_code":"SFO","to_airport_code":"SEA","departure_date":"2026-04-14","passengers":1},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=None)),
            ]
            vdb.add(kind="case",
                    text=json.dumps({"signature": case3_sig.to_text(), "plan": {"steps": case3_steps}}),
                    meta={"destination": "San Francisco"})

            # Seed Case 4: Day trip to Lima (same day return, no hotel)
            case4_sig = Signature(domain="travel", goal="business meeting same-day return", destination="Lima",
                                party={"travelers":3},
                                constraints={"budget":"mid"},
                                preferences={"quick_turnaround":True},
                                activities=[
                                    {"type": "meeting", "description": "client meeting", "location": "Downtown Lima Conference Center"}
                                ],
                                extras={"season":"all_year", "origin":"Sao Paulo", "purpose":"client_meeting", "trip_type":"day_trip", "venue":"Downtown Lima Conference Center"})
            case4_steps = [
                asdict(PlanStep(step_id=0, title="Sao Paulo to Lima morning flight",
                                action="quote_flight",
                                inputs={"from_airport_code":"GRU","to_airport_code":"LIM","departure_date":"2026-06-15","passengers":3},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=1)),
                asdict(PlanStep(step_id=1, title="Lima to Sao Paulo evening flight",
                                action="quote_flight",
                                inputs={"from_airport_code":"LIM","to_airport_code":"GRU","departure_date":"2026-06-15","passengers":3},
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[],
                                next_step=None)),
            ]
            vdb.add(kind="case",
                    text=json.dumps({"signature": case4_sig.to_text(), "plan": {"steps": case4_steps}}),
                    meta={"destination": "Lima"})

        # Facts
        vdb.add(kind="fact",
                text="Midtown Manhattan has many business hotels with conference facilities and excellent transit access.",
                meta={"destination":"NYC"})
        vdb.add(kind="fact",
                text="In NYC winters, walking between meetings can be challenging; consider booking hotels near venues.",
                meta={"destination":"NYC"})
        vdb.add(kind="fact",
                text="JFK and LaGuardia airports serve NYC; JFK is better for international and long-distance flights.",
                meta={"destination":"NYC"})
        vdb.add(kind="fact",
                text="Washington DC has a well-developed Metro system connecting airports to downtown business districts.",
                meta={"destination":"Washington DC"})
        vdb.add(kind="fact",
                text="San Francisco is hilly; efficient routes should avoid steep areas like Russian Hill and Nob Hill.",
                meta={"destination":"San Francisco"})
        vdb.add(kind="fact",
                text="SOMA District in San Francisco is the tech hub with many conference venues and coworking spaces.",
                meta={"destination":"San Francisco"})
        vdb.add(kind="fact",
                text="Lima Peru is reachable from Sao Paulo in about 5 hours; ideal for same-day business trips.",
                meta={"destination":"Lima"})
        vdb.add(kind="fact",
                text="For day trips, book morning outbound and evening return flights to maximize time at destination.",
                meta={})

        # Skills (semantic hints)
        vdb.add(kind="skill",
                text="Hotel selection near business districts; prefer Midtown Manhattan for business travelers visiting NYC.",
                meta={"destination":"NYC"})
        vdb.add(kind="skill",
                text="Bike rental can be useful for quick transportation between nearby meetings in urban areas.",
                meta={"destination":"NYC"})
        vdb.add(kind="skill",
                text="Multi-city trips require connecting flights or inter-city transportation between destinations.",
                meta={})
        vdb.add(kind="skill",
                text="For business travelers, prioritize central locations with good transit access and minimize travel time.",
                meta={})
        vdb.add(kind="skill",
                text="Check signature.activities array for requested services like tours, restaurants, day trips. Create plan steps for each activity.",
                meta={})
        vdb.add(kind="skill",
                text="Activities should be inserted between hotel check-in and departure, in chronological order based on dates and timing.",
                meta={})

        return PlanAgent(vdb=vdb, llm=llm, action_catalog=action_catalog, prompts=prompts)


    
    
    
    def run(self, payload):
        # Initialize a new request context
        function = 'run > generate_plan'
        context = RequestContext()
        
        if 'portfolio' in payload:
            context.portfolio = payload['portfolio']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No portfolio provided'}
        
        if 'org' in payload:
            context.org = payload['org']
        else:
            return {'success':False,'function':function,'input':payload,'output':'No org provided'}
        
        try:
            self._set_context(context)
            
            # Create AgentUtilities with proper config
            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                'some_entity_type',
                'some_entity_id',
                'some_thread'
            )
            
            
            results = []
            print('Initializing PES>GeneratePlan')
            agent = self.build_plan_generator(
                portfolio=context.portfolio,
                org=context.org,
                prompt_ring="pes_prompts",  # Can be made configurable
                action_ring="schd_actions",  # Can be made configurable
                case_ring="pes_cases"  # Can be made configurable
            )
            print('Finished building Plan Generator')
            
            req = {
            "request":payload["message"]
            }
            
            
            response_1 = agent.propose(req)
            results.append(response_1)
            if not response_1['success']: 
                return {'success': False, 'output': results}
            
            # All went well, report back
            return {'success': True, 'message': 'run completed', 'input': payload, 'output': results}
            
        except Exception as e:
            print(f'Error during execution: {str(e)}')
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@generate_plan/run: {str(e)}'}

# Test block
if __name__ == '__main__':
    # Creating an instance
    handler = GeneratePlan()
    '''req = {
        "portfolio":"12345",
        "org":"56789",
        "message":"A team of four executives need to fly to Lima Peru from Sao Paulo in the morning and come back the same day."
    }'''
    
    req = {
        "portfolio":"12345",
        "org":"56789",
        "message":"A team of three executives are going to New York City from Rio de Janeiro on November 15 they are staying for 4 nights and then go to Chicago for 3 nights. Then fly back to Rio"
    }
    
    req_x = {
        "portfolio":"12345",
        "org":"56789",
        "message":"A group of 42 people are coming to NYC in September 17 from Milan Italy for four nights, they want a city tour, a dinner in a good restaurant. Then they want to have a day trip to Philadelphia. The inbound flight arrives at 2:30PM. "
    }
    
    out = handler.run(req)
    print(json.dumps(out, indent=2, cls=DecimalEncoder))
    
