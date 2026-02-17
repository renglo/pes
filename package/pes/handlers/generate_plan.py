from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union
import json, math, time, uuid, re
import os
from openai import OpenAI
from collections import Counter
from decimal import Decimal
from datetime import datetime
from zoneinfo import ZoneInfo

from renglo.agent.agent_utilities import AgentUtilities
from renglo.common import load_config
from renglo.data.data_controller import DataController
from renglo.blueprint.blueprint_controller import BlueprintController

from contextvars import ContextVar


# ────────────────────────────────────────────────────────────────────────────────
# Intent schema (PES domain-agnostic; single source of truth for request characteristics)
# ────────────────────────────────────────────────────────────────────────────────


def new_intent(intent_id: str, user_message: str) -> Dict[str, Any]:
    """Create a fresh intent template. Schema: renglo.intent.v1"""
    now = int(time.time())
    return {
        "schema": "renglo.intent.v1",
        "intent_id": intent_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "user_message": user_message,
            "locale": "en-US",
            "timezone": "America/New_York",
            "now_iso": None,
            "now_date": None,
        },
        "status": {
            "phase": "intake",
            "state": "collecting_requirements",
            "missing_required": [],
            "assumptions": [],
            "notes": [],
        },
        "party": {
            "travelers": {"adults": 0, "children": 0, "infants": 0},
            "traveler_profile_ids": [],
            "guests": [],
            "contact": {"email": None, "phone": None},
        },
        "itinerary": {
            "trip_type": None,
            "segments": [],
            "lodging": {
                "needed": True,
                "check_in": None,
                "check_out": None,
                "location_hint": None,
                "stays": [],
            },
            "ground": {"needed": False},
        },
        "preferences": {"flight": {}, "hotel": {}},
        "constraints": {"budget_total": None, "currency": "USD", "refundable_preference": "either"},
        "policy": {"rules": {"require_user_approval_to_purchase": True, "holds_allowed_without_approval": True}},
        "working_memory": {
            "flight_quotes": [],
            "hotel_quotes": [],
            "flight_quotes_by_segment": [],
            "hotel_quotes_by_stay": [],
            "ranked_bundles": [],
            "risk_report": None,
            "selected": {"bundle_id": None, "flight_option_id": None, "hotel_option_id": None, "flight_option_ids": [], "hotel_option_ids": []},
            "holds": [],
            "bookings": [],
        },
        "audit": {"events": []},
    }


def intent_to_text(ti: Dict[str, Any]) -> str:
    """Compact JSON for retrieval and plan generation. Extracts key request characteristics."""
    out: Dict[str, Any] = {}
    party = ti.get("party") or {}
    travelers = party.get("travelers") or {}
    if travelers:
        out["party"] = {"travelers": travelers}
    iti = ti.get("itinerary") or {}
    if iti.get("trip_type"):
        out["trip_type"] = iti["trip_type"]
    segs = iti.get("segments") or []
    if segs:
        out["segments"] = [
            {
                "origin": (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin"),
                "destination": (s.get("destination") or {}).get("code") if isinstance(s.get("destination"), dict) else s.get("destination"),
                "depart_date": s.get("depart_date"),
            }
            for s in segs
        ]
    lod = iti.get("lodging") or {}
    if lod.get("stays"):
        out["stays"] = lod["stays"]
    elif lod.get("check_in") or lod.get("check_out"):
        out["lodging"] = {"check_in": lod.get("check_in"), "check_out": lod.get("check_out"), "location_hint": lod.get("location_hint")}
    prefs = ti.get("preferences") or {}
    if prefs:
        out["preferences"] = prefs
    constraints = ti.get("constraints") or {}
    if constraints and any(v for v in constraints.values() if v):
        out["constraints"] = constraints
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def intent_destination(ti: Dict[str, Any]) -> Optional[str]:
    """Extract primary destination for retrieval filters."""
    segs = (ti.get("itinerary") or {}).get("segments") or []
    if segs:
        d = segs[0].get("destination")
        if isinstance(d, dict):
            return d.get("code")
        return d if isinstance(d, str) else None
    lod = (ti.get("itinerary") or {}).get("lodging") or {}
    stays = lod.get("stays") or []
    if stays:
        return stays[0].get("location_code")
    return lod.get("location_hint")


def intent_activities(ti: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract activities for plan selection (from extras or inferred from itinerary)."""
    extras = ti.get("extras") or {}
    acts = extras.get("activities") or []
    if acts:
        return acts
    out: List[Dict[str, Any]] = []
    segs = (ti.get("itinerary") or {}).get("segments") or []
    if segs:
        out.append({"type": "flight", "description": "Flight segments"})
    lod = (ti.get("itinerary") or {}).get("lodging") or {}
    if lod.get("needed") or lod.get("stays"):
        out.append({"type": "lodging", "description": "Hotel stay"})
    return out


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

@dataclass
class Case:
    id: str
    intent_text: str
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
        if task == "TO_TRIP_INTENT":
            return {
                "name": "TripIntentExtract",
                "schema": {
                    "type": "object",
                    "properties": {
                        "origin": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "destination": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "trip_type": {"anyOf": [{"type": "string", "enum": ["one_way", "round_trip", "multi_city"]}, {"type": "null"}]},
                        "dates": {
                            "type": "object",
                            "properties": {
                                "departure_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "return_date": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                            },
                            "additionalProperties": False
                        },
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "origin": {"type": "string"},
                                    "destination": {"type": "string"},
                                    "depart_date": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
                        },
                        "stays": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "location_code": {"type": "string"},
                                    "check_in": {"type": "string"},
                                    "check_out": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
                        },
                        "travelers": {
                            "type": "object",
                            "properties": {
                                "adults": {"type": "integer"},
                                "children": {"type": "integer"},
                                "infants": {"type": "integer"}
                            },
                            "additionalProperties": False
                        },
                        "lodging": {
                            "type": "object",
                            "properties": {
                                "needed": {"type": "boolean"},
                                "check_in": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "check_out": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                            },
                            "additionalProperties": False
                        },
                        "extras": {"type": "object", "additionalProperties": True}
                    },
                    "additionalProperties": True
                },
                "strict": False
            }
        if task in ("ADAPT_PLAN", "COMPOSE_PLAN", "INCREMENT_PLAN"):
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
# Planner
# ────────────────────────────────────────────────────────────────────────────────

class Planner:
    """
    v2.2:
      - LLM-crafted trip intent, plan adaptation/composition, and selection
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
            'intent_text', 'sig_text', 'example_cases', 'fact_texts', 'catalog_summary',
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

    # 1) Trip intent extraction via LLM
    def to_intent(self, req: Dict[str, Any], intent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Extract requirements from user message and return full intent."""
        print('Extracting trip intent from user request...')
        user_message = req.get("request", "") if isinstance(req.get("request"), str) else json.dumps(req)
        if isinstance(user_message, dict):
            user_message = req.get("message", "") or json.dumps(req)
        user_message = str(user_message).strip() if user_message else ""
        print(f'[DEBUG] User message: {user_message}')

        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = ZoneInfo("America/New_York")
        now_dt = datetime.now(tz)
        now_iso = now_dt.isoformat()
        now_date = now_dt.strftime("%Y-%m-%d")

        intent = new_intent(intent_id or f"pes_{uuid.uuid4().hex[:12]}", user_message)
        intent["request"]["now_iso"] = now_iso
        intent["request"]["now_date"] = now_date

        prompt_template = self.prompts.get('to_intent', '')
        if prompt_template:
            replacements = {'request_text': user_message, 'current_time': now_iso, 'now_date': now_date}
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            prompt = f"""You are a travel requirements extractor. Extract trip details from the user message.

Time context: Today is {now_date} ({now_iso}). Use YYYY-MM-DD for all dates. Dates must be on or after today.

Rules:
- Origin/destination: use IATA airport codes when possible (JFK, EWR, SFO, LAX, MIA, MCO, DFW, etc.)
- travelers: {{"adults": N, "children": 0, "infants": 0}} - adults required
- Multi-city: output "segments" (one per flight leg) and "stays" (one per city). First segment origin = departure city.
- For "X days in Y" or "3 nights": check_out = check_in + nights; next stay check_in = previous check_out
- trip_type: "one_way" | "round_trip" | "multi_city"

TASK: TO_TRIP_INTENT
Return ONLY valid JSON:
{{
  "origin": "IATA or null",
  "destination": "IATA or null",
  "trip_type": "one_way|round_trip|multi_city or null",
  "dates": {{"departure_date": "YYYY-MM-DD or null", "return_date": "YYYY-MM-DD or null"}},
  "segments": [{{"origin": "IATA", "destination": "IATA", "depart_date": "YYYY-MM-DD"}}],
  "stays": [{{"location_code": "IATA", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD"}}],
  "travelers": {{"adults": 1, "children": 0, "infants": 0}},
  "lodging": {{"needed": true, "check_in": "YYYY-MM-DD or null", "check_out": "YYYY-MM-DD or null"}},
  "extras": {{}}
}}

User message: {user_message}
"""
        data = self.llm.complete_json(prompt)
        if not data:
            print('[ERROR] LLM returned no data for intent')
            return None

        print('[DEBUG] LLM extraction:', json.dumps(data, indent=2))
        extracted = self._merge_extract_into_intent(intent, data, now_date)
        return extracted

    def _merge_extract_into_intent(self, intent: Dict[str, Any], extracted: Dict[str, Any], now_date: str) -> Dict[str, Any]:
        """Merge LLM-extracted fields into full intent."""
        def clamp_date(d: Optional[str]) -> Optional[str]:
            if not d or not isinstance(d, str) or len(d) != 10:
                return d
            return d if d >= now_date else now_date

        travelers = extracted.get("travelers")
        if isinstance(travelers, dict) and travelers:
            t = dict(travelers)
            for key in ("adults", "children", "infants"):
                if key in t and t[key] is not None:
                    try:
                        t[key] = int(t[key])
                    except (TypeError, ValueError):
                        t[key] = 1 if key == "adults" else 0
            intent.setdefault("party", {})["travelers"] = t

        origin = extracted.get("origin")
        dest = extracted.get("destination")
        dates = extracted.get("dates") or {}
        extracted_segs = extracted.get("segments") or []
        lod = extracted.get("lodging") or {}
        trip_type = extracted.get("trip_type")

        if trip_type:
            intent.setdefault("itinerary", {})["trip_type"] = trip_type

        segs: List[Dict[str, Any]] = []
        if extracted_segs:
            for i, es in enumerate(extracted_segs):
                o = es.get("origin") or es.get("origin_code")
                d = es.get("destination") or es.get("destination_code")
                o_code = o if isinstance(o, str) else (o.get("code") if isinstance(o, dict) else None)
                d_code = d if isinstance(d, str) else (d.get("code") if isinstance(d, dict) else None)
                depart = es.get("depart_date")
                if depart:
                    depart = clamp_date(depart)
                if o_code and d_code and depart:
                    segs.append({
                        "segment_id": es.get("segment_id") or f"seg_{i}",
                        "origin": {"type": "airport", "code": o_code},
                        "destination": {"type": "airport", "code": d_code},
                        "depart_date": depart,
                        "transport_mode": es.get("transport_mode", "flight"),
                        "depart_time_window": {"start": None, "end": None},
                    })
        elif origin and dest:
            dep_date = clamp_date(dates.get("departure_date"))
            segs.append({
                "segment_id": "seg_outbound",
                "origin": {"type": "airport", "code": origin},
                "destination": {"type": "airport", "code": dest},
                "depart_date": dep_date,
                "transport_mode": "flight",
                "depart_time_window": {"start": None, "end": None},
            })
            if trip_type == "round_trip" and dates.get("return_date"):
                ret_date = clamp_date(dates["return_date"])
                segs.append({
                    "segment_id": "seg_return",
                    "origin": {"type": "airport", "code": dest},
                    "destination": {"type": "airport", "code": origin},
                    "depart_date": ret_date,
                    "transport_mode": "flight",
                    "depart_time_window": {"start": None, "end": None},
                })

        if segs:
            intent.setdefault("itinerary", {})["segments"] = segs

        extracted_stays = extracted.get("stays") or lod.get("stays") or []
        if extracted_stays:
            stays = []
            for s in extracted_stays:
                ci = clamp_date(s.get("check_in"))
                co = clamp_date(s.get("check_out"))
                loc = s.get("location_code") or s.get("destination")
                if ci and co and loc:
                    stays.append({"location_code": loc, "check_in": ci, "check_out": co, "location_hint": s.get("location_hint")})
            if stays:
                intent.setdefault("itinerary", {}).setdefault("lodging", {})["stays"] = stays
        elif lod.get("check_in") or lod.get("check_out"):
            ci = clamp_date(lod.get("check_in"))
            co = clamp_date(lod.get("check_out"))
            dest_code = (segs[0].get("destination") or {}).get("code") if segs else dest
            if ci and co and dest_code:
                intent.setdefault("itinerary", {}).setdefault("lodging", {})["stays"] = [
                    {"location_code": dest_code, "check_in": ci, "check_out": co}
                ]

        if "needed" in lod:
            intent.setdefault("itinerary", {}).setdefault("lodging", {})["needed"] = lod["needed"]

        if extracted.get("extras"):
            intent["extras"] = extracted["extras"]

        intent["updated_at"] = int(time.time())
        return intent

    # 2) Retrieval via VectorDB
    def retrieve(self, intent: Dict[str, Any], k_cases: int = 4, k_facts: int = 4, k_skills: int = 6):
        print('Retrieving cases, facts and skills from agent experience...')
        query = intent_to_text(intent)
        dest = intent_destination(intent)
        filt = {k: v for k, v in {"destination": dest}.items() if v}
        cases = self.vdb.search(query=query, kind="case", k=k_cases, filters=filt)
        facts = self.vdb.search(query=query, kind="fact", k=k_facts, filters=filt)
        skills = self.vdb.search(query=query, kind="skill", k=k_skills, filters=filt)

        # LLM skill scoring
        prompt_scores = f"""
            You are a skill ranker.
            TASK: SKILL_SCORING
            Given an intent and a list of candidate skills, score each skill's applicability in [0,1].
            Return JSON: {{"scores": [{{"skill_id": str, "score": float}}]}}

            Intent:
            ```json
            {query}
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
    def adapt_from_cases(self, cases: List[Case], intent: Dict[str, Any], facts: List[VDBItem]) -> Plan:
        print(f'Adapting plan from {len(cases)} example cases...')
        
        # Convert cases to example format
        example_cases = []
        for case in cases[:3]:  # Use up to 3 example cases
            example_cases.append({
                "intent": case.intent_text,
                "plan_steps": [asdict(s) for s in case.plan.steps]
            })
        
        fact_texts = [{"text": f.text, "meta": f.meta} for f in facts]
        catalog_summary = [{"name": t.key, "required_args": t.required_args} for t in self.action_catalog]
        intent_text = intent_to_text(intent)
        
        prompt_template = self.prompts.get('adapt_plan', '')
        if prompt_template:
            example_cases_json = json.dumps(example_cases, indent=2)
            fact_texts_json = json.dumps(fact_texts, indent=2)
            catalog_summary_json = json.dumps(catalog_summary, indent=2)
            plan_id = uuid.uuid4().hex[:8]
            
            
            replacements = {
                'intent_text': intent_text,
                'sig_text': intent_text,
                'example_cases': example_cases_json,
                'fact_texts': fact_texts_json,
                'catalog_summary': catalog_summary_json,
                'plan_id': plan_id
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            prompt = f"""
            You are a planning agent using case-based reasoning.
            TASK: ADAPT_PLAN
            
            INSTRUCTIONS:
            1. Study the EXAMPLE CASES below to understand planning patterns and structure
            2. Analyze the NEW REQUEST (intent) to identify all facts and requirements
            3. Generate a COMPLETE NEW plan by adapting the example patterns to match the new request's facts
            
            ADAPTATION PROCESS:
            - Extract the structure and sequence from example cases
            - Replace example values with facts from the new intent
            - Ensure all requirements from the intent are addressed
            - Maintain logical flow and dependencies from examples
            - Adapt step sequences to match the new request's specifics
            
            INCORPORATING INTENT FACTS:
            - Use all information from the intent (segments, party.travelers, stays, preferences, etc.)
            - Check itinerary.segments for flight legs; itinerary.lodging.stays for hotel stays
            - For each activity, create appropriate plan steps using available actions
            - Use dates from segments and stays for timing
            - Use party.travelers for quantities and participant details
            - Apply constraints and preferences from the intent
            - Place activities in chronological order based on dates
            
            NEW REQUEST (intent - use these details for all inputs):
            ```json
            {intent_text}
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
            
            FINAL VALIDATION BEFORE RETURNING:
            - Does the plan address ALL requirements from the trip intent?
            - Are itinerary.segments and lodging.stays represented in plan steps?
            - Are dates and timing constraints respected?
            - Is the plan logically complete and executable?
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
    def compose_from_skills(self, intent: Dict[str, Any], skills: List[VDBItem]) -> Plan:
        print('Composing a new plan from scratch based on the user request...')
        catalog = [{
            "name": t.key,
            "description": t.description,
            "required_args": t.required_args,
            "optional_args": t.optional_args,
            "success_criteria_hint": t.success_criteria_hint
        } for t in self.action_catalog]
        intent_text = intent_to_text(intent)
        
        prompt_template = self.prompts.get('compose_plan', '')
        if prompt_template:
            replacements = {
                'intent_text': intent_text,
                'sig_text': intent_text,
                'catalog': json.dumps(catalog, indent=2),
                'skills': json.dumps([{"id": s.id, "text": s.text, "meta": s.meta} for s in skills], indent=2),
                'plan_id': uuid.uuid4().hex[:8]
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            prompt = f"""
            You are a planner that composes a COMPLETE executable plan using ONLY the allowed actions.
            TASK: COMPOSE_PLAN
            
            INSTRUCTIONS:
            - Analyze the intent to understand all requirements and facts
            - Use the ACTION CATALOG to determine available actions and their required inputs
            - Reference the SKILLS for semantic hints and best practices
            - Create a complete plan that addresses all intent requirements
            
            PLAN COMPOSITION:
            - Start with initial steps that establish prerequisites
            - Add steps for each activity or requirement from the intent
            - Ensure itinerary.segments and lodging.stays are represented as plan steps
            - Respect dates, timing, and sequencing from segments and stays
            - Apply constraints and preferences from the intent
            - Create a logical flow where steps build upon each other
            
            USING THE ACTION CATALOG:
            - Each action has required_args that MUST be provided in step.inputs
            - Optional_args can be included if relevant
            - Use action.success_criteria_hint as a guide for step.success_criteria
            - Ensure all action inputs are populated with values from the intent
            
            INCORPORATING INTENT INFORMATION:
            - Use itinerary.segments for flight legs (origin, destination, depart_date)
            - Use itinerary.lodging.stays for hotel stays (location_code, check_in, check_out)
            - Use party.travelers for passenger counts
            - Use preferences and constraints to refine steps
            - Use extras for domain-specific requirements
            
            Intent:
            ```json
            {intent_text}
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
            - title: Clear, descriptive title for what this step accomplishes
            - action: Must match an action name from the ACTION_CATALOG
            - inputs: Object containing all required_args from the action spec, plus any optional_args
            - enter_guard: Boolean expression (typically "True")
            - success_criteria: Expression to evaluate whether the step succeeded
            
            CHRONOLOGICAL ORDER (CRITICAL):
            - Steps MUST be listed in the order they occur in time
            - step_id values MUST be sequential: 0, 1, 2, 3...
            - next_step links to the following step (step 0 → 1 → 2 → 3 → null)
            - Ensure dependencies are satisfied (depends_on steps execute before dependent steps)
            
            FINAL VALIDATION BEFORE RETURNING:
            - Does the plan address ALL requirements from the trip intent?
            - Are itinerary.segments and lodging.stays represented in plan steps?
            - Are all action required_args provided in step inputs?
            - Is the plan logically complete and executable?
            - Do step dependencies form a valid execution order?
        """
        data = self.llm.complete_json(prompt)
        if not data or "plan" not in data:
            print('[ERROR] LLM returned invalid plan structure')
            return Plan(id=uuid.uuid4().hex[:8], steps=[], meta={"strategy": "compose", "error": "LLM returned invalid plan"})

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
    def select_plan(self, intent: Dict[str, Any], candidates: List[Plan]) -> Plan:
        print('Selecting best plan from list of candidates...')
        
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
        
        activities_requested = intent_activities(intent)
        
        intent_text = intent_to_text(intent)
        prompt_template = self.prompts.get('select_best_plan', '')
        if prompt_template:
            replacements = {
                'intent_text': intent_text,
                'sig_text': intent_text,
                'activities_requested': json.dumps(activities_requested, indent=2),
                'plan_details': json.dumps(plan_details, indent=2)
            }
            prompt = self._replace_tokens(prompt_template, replacements)
        else:
            prompt = f"""
            You are a meta-planner evaluating candidate plans for completeness and quality.
            TASK: SELECT_BEST_PLAN
            
            Given an intent with requirements and multiple candidate plans, choose the best one.
            
            EVALUATION CRITERIA (in order of importance):
            1. REQUIREMENT COVERAGE: Does the plan address ALL requirements from the intent?
               - Check itinerary.segments and lodging.stays - each should have corresponding plan steps
               - Check constraints - all constraints should be respected
               - Check preferences - preferences should be considered
               - Missing requirements = incomplete plan
            
            2. COMPLETENESS: Is the plan logically complete?
               - All necessary steps are present
               - The plan achieves the stated goal
               - No critical gaps in the execution sequence
            
            3. LOGICAL SEQUENCE: Are steps in correct chronological and dependency order?
               - Steps follow a logical progression
               - Dependencies are properly established
               - Timing constraints are respected
            
            4. EFFICIENCY: Reasonable number of steps (not too sparse, not redundant)
               - Plan is neither overly complex nor overly simplistic
               - Steps are appropriately granular
            
            5. ACTION VALIDITY: Are all steps using valid actions with correct inputs?
               - Actions exist in the action catalog
               - Required arguments are provided
               - Input values are appropriate
            
            Return ONLY JSON: {{"choice_index": int, "reason": str}}
            
            The reason should explicitly state:
            - Which requirements are covered/missing in the chosen plan
            - Why this plan is better than the alternatives
            - Any notable strengths or weaknesses

            Intent (with requirements):
            ```json
            {intent_text}
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
            - Prefer plans that cover ALL requirements over those that skip some
            - Consider both completeness and quality
            - Choose the plan most likely to successfully achieve the goal
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
    def propose(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        function = 'propose'
        print('Initiating planning process from intent...')
        print('Intent (preview):', intent_to_text(intent)[:200])
        
        retrieved = self.retrieve(intent)
        if not retrieved:
            return {'success': False, 'function': function, 'input': intent, 'output': 'Could not retrieve'}
        
        cases = retrieved['cases']
        facts = retrieved['facts']
        skills = retrieved['skills']

        candidate_plans: List[Plan] = []
        if cases:
            case_objects = [self._vdb_case_to_case(c) for c in cases[:3]]
            adapted = self.adapt_from_cases(case_objects, intent, facts)
            candidate_plans.append(adapted)
        composed = self.compose_from_skills(intent, skills[:4])
        candidate_plans.append(composed)

        final = self.select_plan(intent, candidate_plans)

        return {
            "function": "propose",
            "success": True,
            "input": intent,
            "output": {
                "intent": intent,
                "plan": asdict(final),
            }
        }

    # Helper: convert VDB 'case' item to Case object
    def _vdb_case_to_case(self, item: VDBItem) -> Case:
        try:
            obj = json.loads(item.text)
            steps = [PlanStep(**s) for s in obj["plan"]["steps"]]
            p = Plan(id=f"plan.from_case.{item.id}", steps=steps, meta={"strategy":"retrieved"})
            intent_txt = obj.get("intent", obj.get("trip_intent", ""))
            return Case(id=item.id, intent_text=intent_txt, plan=p, outcomes={"rating":4.2}, context=item.meta)
        except Exception:
            return Case(id=item.id, intent_text="", plan=Plan(id=f"plan.empty.{item.id}", steps=[], meta={}), outcomes={}, context=item.meta)



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
    
    def _load_prompts(self, portfolio: str, org: str, prompt_ring: str = "pes_prompts", case_group: str = None) -> Dict[str, str]:
        """
        Load prompts from database.
        Returns a dictionary with keys: 'to_intent', 'adapt_plan', 'compose_plan', 'select_best_plan'
        """
        prompts = {
            'to_intent': '',
            'adapt_plan': '',
            'compose_plan': '',
            'select_best_plan': ''
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
                
                #Filter out
                
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
    
    def _load_seed_cases(self, portfolio: str, org: str, case_ring: str = "pes_seed_cases", case_group: str = None) -> List[Dict[str, Any]]:
        """
        Load seed cases from database (trips/cases ring).
        Returns a list of case dictionaries with 'intent' and 'plan' keys.
        """
        cases = []
        
        try:
            # Get all case records from the ring
            if not case_group:
                raise Exception('No case group')
            
            query = {
                'portfolio': portfolio,
                'org': org,
                'ring': case_ring,
                'value': case_group,
                'limit': 99,
                'operator': 'begins_with',
                'lastkey': None,
                'sort': 'asc'
            }
            response = self.DAC.get_a_b_query(query)
            if response and 'items' in response:
                for item in response['items']:
                    # Expect case records to have 'intent' and 'plan' fields (trip_intent for backward compat)
                    intent_text = item.get('intent', item.get('trip_intent', ''))
                    plan_data = item.get('plan', {})
                    
                    if intent_text and plan_data:
                        cases.append({
                            'intent': intent_text,
                            'plan': plan_data
                        })
        except Exception as e:
            print(f'Warning: Could not load seed cases from database: {str(e)}')
        
        return cases
    
    def _load_facts(self, portfolio: str, org: str, fact_ring: str = "pes_seed_facts", case_group: str = None) -> List[Dict[str, Any]]:
        """
        Load facts from database (facts ring).
        Returns a list of fact dictionaries with 'text' and 'meta' keys.
        """
        facts = []
        
        try:
            # Get all fact records from the ring
            if not case_group:
                raise Exception('No case group')
            
            query = {
                'portfolio': portfolio,
                'org': org,
                'ring': fact_ring,
                'value': case_group,
                'limit': 99,
                'operator': 'begins_with',
                'lastkey': None,
                'sort': 'asc'
            }
            response = self.DAC.get_a_b_query(query)
            if response and 'items' in response:
                for item in response['items']:
                    # Expect fact records to have 'text' and 'meta' fields
                    text = item.get('text', '')
                    meta = item.get('meta', {})
                    
                    if text:
                        facts.append({
                            'text': text,
                            'meta': meta
                        })
        except Exception as e:
            print(f'Warning: Could not load facts from database: {str(e)}')
        
        return facts
    
    
    def build_plan_generator(self, portfolio: str, org: str, 
                         prompt_ring: str = "pes_prompts",
                         action_ring: str = "schd_actions",
                         case_ring: str = "pes_cases",
                         fact_ring: str = "pes_facts",
                         plan_actions: Union[str, List[str]] = "") -> Planner:
        
        embedder = SimpleEmbedder()
        vdb = VectorDB(embedder)
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
        
        # Filter action catalog to only include actions specified in plan_actions
        # plan_actions can be a list of strings or a comma-separated string (for backward compatibility)
        action_catalog_specific = []
        if plan_actions:
            # Normalize to a set: handle both list and comma-separated string formats
            if isinstance(plan_actions, list):
                plan_actions_set = {action.strip() for action in plan_actions if action.strip()}
            else:
                # Backward compatibility: parse comma-separated string
                plan_actions_set = {action.strip() for action in plan_actions.split(',') if action.strip()}
            
            for a in action_catalog:
                if a.key in plan_actions_set:
                    action_catalog_specific.append(a)
  
        # Note: If no actions are loaded from database, action_catalog will be empty
        if not action_catalog_specific:
            print('Warning: No actions loaded from database, action_catalog_specific will be empty')

        # Load seed cases from database
        seed_cases = self._load_seed_cases(portfolio, org, case_ring, case_group=case_group)
        
        # Add seed cases to VectorDB
        for case_data in seed_cases:
            intent_text = case_data.get('intent', case_data.get('trip_intent', ''))
            plan_data = case_data.get('plan', {})
            meta = case_data.get('meta', {})
            
            if intent_text and plan_data:
                vdb.add(kind="case",
                       text=json.dumps({"intent": intent_text, "plan": plan_data}),
                       meta=meta)
        
        # Note: If no seed cases are loaded from database, VDB will be empty
        if not seed_cases:
            print('Warning: No seed cases loaded from database, VDB will be empty')

        # Load facts from database
        facts = self._load_facts(portfolio, org, fact_ring, case_group=case_group)
        
        # Add facts to VectorDB
        for fact_data in facts:
            text = fact_data.get('text', '')
            kind = fact_data.get('kind', 'fact')
            meta = fact_data.get('meta', {})
            
            if text:
                vdb.add(kind=kind,
                       text=text,
                       meta=meta)
        
        # Note: If no facts are loaded from database, VDB will not have facts
        if not facts:
            print('Warning: No facts loaded from database, VDB will not have facts')

        return Planner(vdb=vdb, llm=llm, action_catalog=action_catalog_specific, prompts=prompts)

    
    def run(self, payload):
        
        '''
        Payload
        {
            "portfolio":"",
            "org":"",
            "message":""
        }
        '''
        # Initialize a new request context
        function = 'run > generate_plan'
        print(f'Running:{function}')
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
        
        if '_init' in payload:
            raw = payload['_init']
            context.init = json.loads(raw) if isinstance(raw, str) else raw
        else:
            context.init = {}
            
        
        try:
            self._set_context(context)
            
            entity_type = payload.get('_entity_type') or 'some_entity_type'
            entity_id = payload.get('_entity_id') or 'some_entity_id'
            thread = payload.get('_thread') or 'some_thread'
            workspace_id = payload.get('workspace_id') or payload.get('workspace')
            public_user = payload.get('public_user')
            
            self.AGU = AgentUtilities(
                self.config,
                context.portfolio,
                context.org,
                entity_type,
                entity_id,
                thread
            )
            print("Agent Utilities initialized")
            
            if 'plan_actions' in context.init:
                plan_actions = context.init['plan_actions']
            else:
                plan_actions = ''
            print(f'Plan Actions loaded:{plan_actions}')

            results = []
            print('Initializing PES>GeneratePlan')
            planner = self.build_plan_generator(
                portfolio=context.portfolio,
                org=context.org,
                prompt_ring="pes_prompts",
                action_ring="schd_actions",
                case_ring="pes_cases",
                plan_actions=plan_actions
            )
            print('Finished building Plan Generator')
            
            user_message = payload.get("message", "").strip()
            req = {"request": user_message, "message": user_message}
            
            intent = planner.to_intent(req)
            if not intent:
                return {'success': False, 'function': function, 'input': payload, 'output': 'Intent could not be extracted'}
            
            self.AGU.mutate_workspace(
                {'request_intent': intent},
                public_user=public_user,
                workspace_id=workspace_id
            )
            print('Intent stored in workspace')
            
            response_1 = planner.propose(intent)
            results.append(response_1)
            if not response_1['success']:
                return {'success': False, 'output': results}
            
            canonical = results[-1]['output']
            return {'success': True, 'interface': 'plan', 'input': payload, 'output': canonical, 'stack': results}
            
        except Exception as e:
            print(f'Error during execution: {str(e)}')
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@generate_plan/run: {str(e)}'}

