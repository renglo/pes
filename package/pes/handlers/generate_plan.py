from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union
import copy
import json
import math
import re
import time
import uuid
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
            "traveler_ids": [],
            "travelers_by_id": {},
            "traveler_profile_ids": [],
            "guests": [],
            "contact": {"email": None, "phone": None},
        },
        "itinerary": {
            "trip_type": None,
            "segments": [],
            "day_trips": [],
            "converging": False,
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


def intent_for_retrieval(ti: Dict[str, Any]) -> Dict[str, Any]:
    """Extract retrieval-relevant fields from intent. Returns dict; serialize with json.dumps() when needed."""
    out: Dict[str, Any] = {}
    party = ti.get("party") or {}
    travelers = party.get("travelers") or {}
    if travelers:
        out["party"] = {"travelers": travelers}
        if party.get("traveler_ids"):
            out["party"]["traveler_ids"] = party["traveler_ids"]
        if party.get("travelers_by_id"):
            out["party"]["travelers_by_id"] = party["travelers_by_id"]
    iti = ti.get("itinerary") or {}
    if iti.get("trip_type"):
        out["trip_type"] = iti["trip_type"]
    if iti.get("converging"):
        out["converging"] = True
    segs = iti.get("segments") or []
    if segs:
        out["segments"] = []
        for s in segs:
            seg = {
                "origin": (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin"),
                "destination": (s.get("destination") or {}).get("code") if isinstance(s.get("destination"), dict) else s.get("destination"),
                "depart_date": s.get("depart_date"),
                "passengers": s.get("passengers"),
            }
            if s.get("traveler_ids"):
                seg["traveler_ids"] = s["traveler_ids"]
            out["segments"].append(seg)
    lod = iti.get("lodging") or {}
    if lod.get("stays"):
        out["stays"] = lod["stays"]
    elif lod.get("check_in") or lod.get("check_out"):
        out["lodging"] = {"check_in": lod.get("check_in"), "check_out": lod.get("check_out"), "location_hint": lod.get("location_hint")}
    if lod.get("number_of_nights") is not None:
        out.setdefault("lodging", {})["number_of_nights"] = lod["number_of_nights"]
    if "needed" in lod:
        out.setdefault("lodging", {})["needed"] = lod["needed"]
    extras = ti.get("extras") or {}
    if extras.get("activities"):
        out["activities"] = extras["activities"]
    if extras.get("itinerary"):
        out["itinerary"] = extras["itinerary"]
    if extras.get("travelers"):
        out["converging_travelers"] = extras["travelers"]
    prefs = ti.get("preferences") or {}
    if prefs:
        out["preferences"] = prefs
    constraints = ti.get("constraints") or {}
    if constraints and any(v for v in constraints.values() if v):
        out["constraints"] = constraints
    return out


def intent_for_plan(ti: Dict[str, Any]) -> Dict[str, Any]:
    """Extract plan-relevant fields from intent for prompts. Returns dict; serialize with json.dumps() when needed."""
    return {
        "party": ti.get("party") or {},
        "itinerary": ti.get("itinerary") or {},
        "extras": ti.get("extras") or {},
        "preferences": ti.get("preferences") or {},
        "constraints": ti.get("constraints") or {},
    }

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
                        "trip_type": {"anyOf": [{"type": "string", "enum": ["one_way", "round_trip", "multi_city", "single_destination", "day_trip", "converging"]}, {"type": "null"}]},
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
                                    "depart_date": {"type": "string"},
                                    "passengers": {"type": "integer"},
                                    "traveler_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "For converging: e.g. [\"t1\",\"t2\",\"t3\"]"
                                    }
                                },
                                "additionalProperties": True
                            }
                        },
                        "stays": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "location_code": {"type": "string"},
                                    "check_in": {"type": "string"},
                                    "check_out": {"type": "string"},
                                    "number_of_guests": {"type": "integer"},
                                    "traveler_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "For converging: e.g. [\"t1\",\"t2\",\"t3\"]"
                                    }
                                },
                                "additionalProperties": True
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
                                "check_out": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                "number_of_nights": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
                            },
                            "additionalProperties": False
                        },
                        "activities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "description": {"type": "string"},
                                    "location": {"type": "string"},
                                    "when": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
                        },
                        "itinerary": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from": {"type": "string"},
                                    "to": {"type": "string"},
                                    "date": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
                        },
                        "converging_travelers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer"},
                                    "origin": {"type": "string"},
                                    "arrival_date": {"type": "string"}
                                },
                                "additionalProperties": False
                            }
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
        if task == "MODIFY_INTENT_DELTA":
            return {
                "name": "ModifyIntentDelta",
                "schema": {
                    "type": "object",
                    "properties": {
                        "changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["extend_stay", "shorten_stay", "add_side_trip", "change_dates"]},
                                    "until_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "add_nights": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                                    "remove_nights": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                                    "city": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "nights": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                                    "departure_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                                    "return_date": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                                },
                                "required": ["type"],
                                "additionalProperties": True
                            }
                        }
                    },
                    "required": ["changes"],
                    "additionalProperties": False
                },
                "strict": False
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
            'intent_text', 'example_cases', 'fact_texts', 'catalog_summary',
            'catalog', 'skills', 'activities_requested', 'plan_details', 'intent_examples'
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

    # 1) Trip intent extraction via LLM - examples inform extraction (intent should be explicit for plan)
    def to_intent(self, req: Dict[str, Any], intent_id: Optional[str] = None,
                  cases: Optional[List[VDBItem]] = None, facts: Optional[List[VDBItem]] = None,
                  skills: Optional[List[VDBItem]] = None) -> Optional[Dict[str, Any]]:
        """Extract requirements from user message. Examples from VDB inform how to structure the intent."""
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

        intent_examples = ""
        if cases:
            examples = []
            for c in cases[:3]:
                try:
                    obj = json.loads(c.text)
                    ex_intent = obj.get("intent", obj.get("trip_intent", ""))
                    if ex_intent:
                        examples.append({"intent_structure": ex_intent})
                except Exception:
                    pass
            if examples:
                intent_examples = json.dumps(examples, indent=2)

        fact_texts = ""
        if facts:
            fact_texts = json.dumps([{"text": f.text, "meta": f.meta} for f in facts[:4]], indent=2)

        prompt_template = self.prompts.get('to_intent', '')
        if prompt_template:
            replacements = {
                'request_text': user_message, 'current_time': now_iso, 'now_date': now_date,
                'intent_examples': intent_examples or '[]', 'fact_texts': fact_texts or '[]'
            }
            prompt = self._replace_tokens(prompt_template, replacements)
            if '#intent_examples#' not in prompt_template and intent_examples:
                prompt += f"\n\nEXAMPLE INTENTS (learn structure from these):\n```json\n{intent_examples}\n```"
        else:
            # FALLBACK: used when pes_prompts.to_intent is empty; domain-specific (trips)
            prompt = f"""You are a travel requirements extractor. Extract trip details from the user message.

Time context: Today is {now_date} ({now_iso}). Use YYYY-MM-DD for all dates. Dates must be on or after today.

TRIP TYPES:
- single_destination: Origin → Destination → Origin with overnight stay (default when origin+dest+lodging)
- round_trip: Same as single_destination (use interchangeably)
- day_trip: Same-day return, NO hotel (origin → dest → origin same day)
- multi_city: A → B → C → ... → A (multiple cities, use "segments" and "itinerary")
- converging: Multiple groups from different origins meeting at one destination (use "converging_travelers")

RULES:
- Origin/destination: use IATA airport codes when possible (JFK, EWR, SFO, LAX, MIA, MCO, DFW, GRU, etc.)
- travelers: {{"adults": N, "children": 0, "infants": 0}} - adults required
- For "X days in Y" or "3 nights": check_out = check_in + nights; lodging.number_of_nights = X
- Multi-city: output "segments" (one per flight leg) and "itinerary" [{{"from":"City","to":"City","date":"YYYY-MM-DD"}}]
- Day trip: same date for outbound and return, lodging.needed = false, no stays
- Activities: Extract ALL requested (city_tour, restaurant, day_trip, theater, museum, spa, conference, meeting)
  Format: [{{"type":"city_tour","description":"guided tour","location":"City"}}]
- Converging: converging_travelers = [{{"count":N,"origin":"City","arrival_date":"YYYY-MM-DD"}}]
  - For converging: create SEPARATE stays per group (each with own check_in/check_out, number_of_guests=count)
  - Each segment and stay MUST have explicit passengers/number_of_guests

TASK: TO_TRIP_INTENT
Return ONLY valid JSON:
{{
  "origin": "IATA or null",
  "destination": "IATA or null",
  "trip_type": "single_destination|round_trip|one_way|day_trip|multi_city|converging or null",
  "dates": {{"departure_date": "YYYY-MM-DD or null", "return_date": "YYYY-MM-DD or null"}},
  "segments": [{{"origin": "IATA", "destination": "IATA", "depart_date": "YYYY-MM-DD", "passengers": N}}],
  "stays": [{{"location_code": "IATA", "check_in": "YYYY-MM-DD", "check_out": "YYYY-MM-DD", "number_of_guests": N}}],
  "travelers": {{"adults": 1, "children": 0, "infants": 0}},
  "lodging": {{"needed": true, "check_in": "YYYY-MM-DD or null", "check_out": "YYYY-MM-DD or null", "number_of_nights": N or null}},
  "activities": [{{"type": "activity_type", "description": "details", "location": "where"}}],
  "itinerary": [{{"from": "City", "to": "City", "date": "YYYY-MM-DD"}}],
  "converging_travelers": [{{"count": N, "origin": "City", "arrival_date": "YYYY-MM-DD"}}],
  "extras": {{}}
}}

User message: {user_message}
"""
            if intent_examples:
                prompt += f"\n\nEXAMPLE INTENTS (learn structure from these similar cases):\n```json\n{intent_examples}\n```"
            if fact_texts:
                prompt += f"\n\nRELEVANT FACTS:\n```json\n{fact_texts}\n```"
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

        def _total_travelers(t: Dict[str, Any]) -> int:
            if not t:
                return 1
            return int(t.get("adults", 0) or 0) + int(t.get("children", 0) or 0) + int(t.get("infants", 0) or 0)

        def _build_traveler_ids() -> Tuple[List[str], List[List[str]], Dict[str, Dict[str, Any]]]:
            """Build traveler_ids, traveler_ids_by_group (for converging), and travelers_by_id."""
            by_id: Dict[str, Dict[str, Any]] = {}
            ids_list: List[str] = []
            ids_by_group: List[List[str]] = []
            if converging:
                idx = 0
                for gi, ct in enumerate(converging):
                    n = int(ct.get("count", 0) or 0)
                    group_ids = []
                    for _ in range(n):
                        tid = f"t{idx + 1}"
                        ids_list.append(tid)
                        group_ids.append(tid)
                        by_id[tid] = {
                            "group_index": gi,
                            "origin": ct.get("origin"),
                            "arrival_date": ct.get("arrival_date"),
                        }
                        idx += 1
                    ids_by_group.append(group_ids)
            else:
                n = _total_travelers(travelers) if isinstance(travelers, dict) else 1
                group_ids = [f"t{i + 1}" for i in range(n)]
                ids_list = group_ids
                ids_by_group = [group_ids]
                for i, tid in enumerate(group_ids):
                    by_id[tid] = {"group_index": 0}
            return ids_list, ids_by_group, by_id

        segs: List[Dict[str, Any]] = []
        converging = extracted.get("converging_travelers") or []
        default_passengers = _total_travelers(travelers) if isinstance(travelers, dict) else 1
        traveler_ids_list, traveler_ids_by_group, travelers_by_id = _build_traveler_ids()
        intent.setdefault("party", {})["traveler_ids"] = traveler_ids_list
        intent.setdefault("party", {})["travelers_by_id"] = travelers_by_id
        if extracted_segs:
            for i, es in enumerate(extracted_segs):
                o = es.get("origin") or es.get("origin_code")
                d = es.get("destination") or es.get("destination_code")
                o_code = o if isinstance(o, str) else (o.get("code") if isinstance(o, dict) else None)
                d_code = d if isinstance(d, str) else (d.get("code") if isinstance(d, dict) else None)
                depart = es.get("depart_date")
                if depart:
                    depart = clamp_date(depart)
                pax = es.get("passengers")
                if pax is None and converging:
                    o_upper = str(o_code).upper() if o_code else ""
                    for ct in converging:
                        orig = (ct.get("origin") or "").upper()
                        if len(orig) == 3 and o_upper == orig:
                            pax = int(ct.get("count", 0) or 0)
                            break
                        if orig and o_upper and orig in o_upper:
                            pax = int(ct.get("count", 0) or 0)
                            break
                if pax is None:
                    pax = default_passengers
                raw_tids = es.get("traveler_ids") if isinstance(es.get("traveler_ids"), list) else []
                valid_tids = set(traveler_ids_list)
                seg_tids = [t for t in raw_tids if t in valid_tids] if raw_tids else []
                if not seg_tids and converging:
                    o_upper = str(o_code).upper() if o_code else ""
                    d_upper = str(d_code).upper() if d_code else ""
                    for gi, ct in enumerate(converging):
                        orig = (ct.get("origin") or "").upper()
                        if (len(orig) == 3 and o_upper == orig) or (orig and o_upper and orig in o_upper):
                            seg_tids = traveler_ids_by_group[gi] if gi < len(traveler_ids_by_group) else []
                            break
                        if (len(orig) == 3 and d_upper == orig) or (orig and d_upper and orig in d_upper):
                            seg_tids = traveler_ids_by_group[gi] if gi < len(traveler_ids_by_group) else []
                            break
                    origins_set = {(ct.get("origin") or "").upper()[:3] for ct in converging if ct.get("origin")}
                    is_meeting_point = o_upper and o_upper not in origins_set and len(o_upper) == 3
                    if not seg_tids and is_meeting_point:
                        seg_tids = traveler_ids_list
                if not seg_tids and not converging:
                    seg_tids = traveler_ids_list
                if o_code and d_code and depart:
                    seg = {
                        "segment_id": es.get("segment_id") or f"seg_{i}",
                        "origin": {"type": "airport", "code": o_code},
                        "destination": {"type": "airport", "code": d_code},
                        "depart_date": depart,
                        "passengers": pax,
                        "transport_mode": es.get("transport_mode", "flight"),
                        "depart_time_window": {"start": None, "end": None},
                    }
                    if seg_tids:
                        seg["traveler_ids"] = seg_tids
                    segs.append(seg)
            # For converging: add return segments (one per group to their origin) if not already present
            if converging and segs:
                dest_code = (segs[0].get("destination") or {}).get("code") if segs else None
                has_return = False
                if dest_code:
                    for s in segs:
                        o = (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin")
                        if str(o or "").upper() == str(dest_code).upper():
                            has_return = True
                            break
                if not has_return and dest_code:
                    ret_date = clamp_date(dates.get("return_date")) or clamp_date(lod.get("check_out"))
                    if not ret_date and (lod.get("stays") or extracted.get("stays")):
                        sts = lod.get("stays") or extracted.get("stays") or []
                        s0 = sts[0] if sts else {}
                        ret_date = clamp_date(s0.get("check_out"))
                    if not ret_date and lod.get("check_in") and lod.get("number_of_nights"):
                        try:
                            from datetime import datetime, timedelta
                            ci = lod.get("check_in")
                            nights = int(lod.get("number_of_nights", 0))
                            if ci and nights:
                                ret_date = (datetime.strptime(str(ci)[:10], "%Y-%m-%d") + timedelta(days=nights)).strftime("%Y-%m-%d")
                        except Exception:
                            pass
                    if ret_date:
                        for i, ct in enumerate(converging):
                            orig = ct.get("origin") or ""
                            orig_code = str(orig).upper() if len(str(orig)) == 3 else str(orig).upper()
                            if len(orig_code) == 3:
                                n_pax = int(ct.get("count", 0) or 0)
                                if n_pax:
                                    seg_tids = traveler_ids_by_group[i] if i < len(traveler_ids_by_group) else []
                                    seg = {
                                        "segment_id": f"seg_return_{i}",
                                        "origin": {"type": "airport", "code": dest_code},
                                        "destination": {"type": "airport", "code": orig_code},
                                        "depart_date": ret_date,
                                        "passengers": n_pax,
                                        "transport_mode": "flight",
                                        "depart_time_window": {"start": None, "end": None},
                                    }
                                    if seg_tids:
                                        seg["traveler_ids"] = seg_tids
                                    segs.append(seg)
        elif origin and dest:
            dep_date = clamp_date(dates.get("departure_date"))
            if not trip_type:
                trip_type = "round_trip"
            if trip_type == "single_destination":
                trip_type = "round_trip"
            intent.setdefault("itinerary", {})["trip_type"] = trip_type
            outbound_seg = {
                "segment_id": "seg_outbound",
                "origin": {"type": "airport", "code": origin},
                "destination": {"type": "airport", "code": dest},
                "depart_date": dep_date,
                "passengers": default_passengers,
                "transport_mode": "flight",
                "depart_time_window": {"start": None, "end": None},
            }
            if traveler_ids_list:
                outbound_seg["traveler_ids"] = traveler_ids_list
            segs.append(outbound_seg)
            if trip_type != "one_way":
                ret_date = clamp_date(dates.get("return_date"))
                if not ret_date and trip_type == "day_trip":
                    ret_date = dep_date
                if not ret_date and lod.get("stays"):
                    s0 = lod["stays"][0] if lod["stays"] else {}
                    ret_date = clamp_date(s0.get("check_out"))
                if not ret_date:
                    ret_date = clamp_date(lod.get("check_out"))
                if not ret_date and lod.get("check_in") and lod.get("number_of_nights"):
                    try:
                        from datetime import datetime, timedelta
                        ci = lod.get("check_in")
                        nights = int(lod.get("number_of_nights", 0))
                        if ci and nights:
                            ci_dt = datetime.strptime(str(ci)[:10], "%Y-%m-%d")
                            ret_date = (ci_dt + timedelta(days=nights)).strftime("%Y-%m-%d")
                    except Exception:
                        pass
                if ret_date:
                    ret_pax = default_passengers
                    if converging:
                        ret_pax = sum(int(ct.get("count", 0) or 0) for ct in converging)
                    ret_seg = {
                        "segment_id": "seg_return",
                        "origin": {"type": "airport", "code": dest},
                        "destination": {"type": "airport", "code": origin},
                        "depart_date": ret_date,
                        "passengers": ret_pax,
                        "transport_mode": "flight",
                        "depart_time_window": {"start": None, "end": None},
                    }
                    if traveler_ids_list:
                        ret_seg["traveler_ids"] = traveler_ids_list
                    segs.append(ret_seg)

        if segs:
            intent.setdefault("itinerary", {})["segments"] = segs

        dest_code = (segs[0].get("destination") or {}).get("code") if segs else dest
        ret_date = clamp_date(dates.get("return_date")) or (clamp_date(lod.get("check_out")) if lod else None)
        if not ret_date and segs:
            for s in segs:
                o = (s.get("origin") or {}).get("code") if isinstance(s.get("origin"), dict) else s.get("origin")
                if str(o).upper() == str(dest_code).upper():
                    ret_date = clamp_date(s.get("depart_date"))
                    break

        if converging and dest_code:
            stays = []
            for i, ct in enumerate(converging):
                arr = clamp_date(ct.get("arrival_date"))
                co = ret_date or clamp_date(lod.get("check_out"))
                if not co and lod.get("check_in") and lod.get("number_of_nights"):
                    try:
                        from datetime import datetime, timedelta
                        ci = lod.get("check_in")
                        nights = int(lod.get("number_of_nights", 0))
                        if ci and nights:
                            co = (datetime.strptime(str(ci)[:10], "%Y-%m-%d") + timedelta(days=nights)).strftime("%Y-%m-%d")
                    except Exception:
                        pass
                n_guests = int(ct.get("count", 0) or 0)
                if arr and co and n_guests:
                    stay_tids = traveler_ids_by_group[i] if i < len(traveler_ids_by_group) else []
                    stay = {
                        "location_code": dest_code,
                        "check_in": arr,
                        "check_out": co,
                        "number_of_guests": n_guests,
                    }
                    if stay_tids:
                        stay["traveler_ids"] = stay_tids
                    stays.append(stay)
            if stays:
                intent.setdefault("itinerary", {}).setdefault("lodging", {})["stays"] = stays
        else:
            extracted_stays = extracted.get("stays") or lod.get("stays") or []
            if extracted_stays:
                stays = []
                for s in extracted_stays:
                    ci = clamp_date(s.get("check_in"))
                    co = clamp_date(s.get("check_out"))
                    loc = s.get("location_code") or s.get("destination")
                    n_guests = s.get("number_of_guests")
                    if n_guests is None:
                        n_guests = default_passengers
                    else:
                        n_guests = int(n_guests)
                    if ci and co and loc:
                        stay = {
                            "location_code": loc,
                            "check_in": ci,
                            "check_out": co,
                            "number_of_guests": n_guests,
                            "location_hint": s.get("location_hint"),
                        }
                        raw_stay_tids = s.get("traveler_ids") if isinstance(s.get("traveler_ids"), list) else []
                        valid_tids = set(traveler_ids_list)
                        stay_tids = [t for t in raw_stay_tids if t in valid_tids] if raw_stay_tids else None
                        if stay_tids:
                            stay["traveler_ids"] = stay_tids
                        elif traveler_ids_list:
                            stay["traveler_ids"] = traveler_ids_list
                        stays.append(stay)
                if stays:
                    intent.setdefault("itinerary", {}).setdefault("lodging", {})["stays"] = stays
            elif lod.get("check_in") or lod.get("check_out"):
                ci = clamp_date(lod.get("check_in"))
                co = clamp_date(lod.get("check_out"))
                if ci and co and dest_code:
                    n_guests = default_passengers
                    stay = {"location_code": dest_code, "check_in": ci, "check_out": co, "number_of_guests": n_guests}
                    if traveler_ids_list:
                        stay["traveler_ids"] = traveler_ids_list
                    intent.setdefault("itinerary", {}).setdefault("lodging", {})["stays"] = [stay]

        if "needed" in lod:
            intent.setdefault("itinerary", {}).setdefault("lodging", {})["needed"] = lod["needed"]
        if lod.get("number_of_nights") is not None:
            intent.setdefault("itinerary", {}).setdefault("lodging", {})["number_of_nights"] = lod["number_of_nights"]

        activities = extracted.get("activities") or []
        if activities:
            intent.setdefault("extras", {})["activities"] = activities

        itinerary_legs = extracted.get("itinerary") or []
        if itinerary_legs and not segs:
            for i, leg in enumerate(itinerary_legs):
                o = leg.get("from") or leg.get("origin")
                d = leg.get("to") or leg.get("destination")
                dt = clamp_date(leg.get("date") or leg.get("depart_date"))
                if o and d and dt:
                    o_code = str(o).upper() if len(str(o)) == 3 else str(o)
                    d_code = str(d).upper() if len(str(d)) == 3 else str(d)
                    seg = {
                        "segment_id": f"seg_{i}",
                        "origin": {"type": "airport", "code": o_code},
                        "destination": {"type": "airport", "code": d_code},
                        "depart_date": dt,
                        "passengers": default_passengers,
                        "transport_mode": "flight",
                        "depart_time_window": {"start": None, "end": None},
                    }
                    if traveler_ids_list:
                        seg["traveler_ids"] = traveler_ids_list
                    segs.append(seg)
            if segs:
                intent.setdefault("itinerary", {})["segments"] = segs
                if not trip_type:
                    intent.setdefault("itinerary", {})["trip_type"] = "multi_city"

        converging = extracted.get("converging_travelers") or []
        if converging:
            intent.setdefault("extras", {})["travelers"] = converging
            intent.setdefault("itinerary", {})["converging"] = True
            if not trip_type:
                intent.setdefault("itinerary", {})["trip_type"] = "converging"

        if trip_type == "day_trip":
            intent.setdefault("itinerary", {}).setdefault("lodging", {})["needed"] = False

        if extracted.get("extras"):
            for k, v in extracted["extras"].items():
                if v is not None:
                    intent.setdefault("extras", {})[k] = v

        intent["updated_at"] = int(time.time())
        return intent

    # 2) Retrieval via VectorDB - used to inform intent extraction (examples inform intent, not plan)
    def retrieve(self, query: Union[str, Dict[str, Any]], k_cases: int = 4, k_facts: int = 4, k_skills: int = 6):
        """Retrieve cases, facts, skills. query can be user message (str) or intent (dict)."""
        print('Retrieving cases, facts and skills from agent experience...')
        if isinstance(query, dict):
            query_str = json.dumps(intent_for_retrieval(query), sort_keys=True, separators=(",", ":"))
            dest = intent_destination(query)
            filt = {k: v for k, v in {"destination": dest}.items() if v}
        else:
            query_str = str(query).strip()
            filt = None
        cases = self.vdb.search(query=query_str, kind="case", k=k_cases, filters=filt)
        facts = self.vdb.search(query=query_str, kind="fact", k=k_facts, filters=filt)
        skills = self.vdb.search(query=query_str, kind="skill", k=k_skills, filters=filt)

        # LLM skill scoring
        prompt_scores = f"""
            You are a skill ranker.
            TASK: SKILL_SCORING
            Given an intent and a list of candidate skills, score each skill's applicability in [0,1].
            Return JSON: {{"scores": [{{"skill_id": str, "score": float}}]}}

            Intent/Request:
            ```json
            {query_str}
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

    def compose_from_skills(self, intent: Dict[str, Any], skills: List[VDBItem]) -> Plan:
        print('Composing a new plan from scratch based on the user request...')
        catalog = [{
            "name": t.key,
            "description": t.description,
            "required_args": t.required_args,
            "optional_args": t.optional_args,
            "success_criteria_hint": t.success_criteria_hint
        } for t in self.action_catalog]
        intent_json = json.dumps(intent_for_plan(intent), indent=2)
        
        prompt_template = self.prompts.get('compose_plan_light', '')
        if prompt_template:
            replacements = {
                'intent_text': intent_json,
                'sig_text': intent_json,
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
            {intent_json}
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



    def _build_plan_from_intent(self, intent: Dict[str, Any], plan_id: str) -> Plan:
        """
        Build plan programmatically from intent. Intent is the source of truth.
        Order: inbound flights → hotels (one per stay) → return flights.
        Extensible: add activities, tours, restaurants when intent has them and actions exist.
        """
        def _code(v) -> str:
            if isinstance(v, dict):
                return v.get("code") or ""
            return str(v) if v else ""

        iti = intent.get("itinerary") or {}
        segs = iti.get("segments") or []
        lod = iti.get("lodging") or {}
        stays_list = lod.get("stays") or []
        party = intent.get("party") or {}
        travelers = party.get("travelers") or {}
        default_pax = int(travelers.get("adults", 0) or 0) + int(travelers.get("children", 0) or 0) + int(travelers.get("infants", 0) or 0) or 1

        dest_code = _code(segs[0].get("destination")) if segs else None
        last_dest = (stays_list[-1].get("location_code") if stays_list else None) or dest_code
        if isinstance(last_dest, dict):
            last_dest = last_dest.get("code") if last_dest else None
        last_dest = str(last_dest or "").upper() or (str(dest_code or "").upper() if dest_code else "")

        inbound: List[Dict] = []
        return_segs: List[Dict] = []
        intermediate: List[Dict] = []
        for seg in segs:
            d = _code(seg.get("destination"))
            o = _code(seg.get("origin"))
            if dest_code and d and str(d).upper() == str(dest_code).upper():
                inbound.append(seg)
            elif last_dest and o and str(o).upper() == str(last_dest).upper():
                return_segs.append(seg)
            else:
                intermediate.append(seg)
        if not inbound and not return_segs and segs:
            inbound = list(segs)

        steps: List[PlanStep] = []
        leg = 0

        for seg in inbound:
            o_code = _code(seg.get("origin"))
            d_code = _code(seg.get("destination"))
            dep = seg.get("depart_date")
            pax = seg.get("passengers", default_pax)
            tids = seg.get("traveler_ids") or []
            if o_code and d_code and dep:
                steps.append(PlanStep(
                    step_id=len(steps),
                    title=f"{o_code} to {d_code} flight",
                    action="quote_flight",
                    inputs={
                        "from_airport_code": o_code,
                        "to_airport_code": d_code,
                        "departure_date": dep,
                        "leg": leg,
                        "passengers": pax,
                        "traveler_ids": tids,
                    },
                    enter_guard="True",
                    success_criteria="len(result) > 0",
                    depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                    next_step=None,
                ))
                leg += 1

        prev_stay_loc = None
        for stay in stays_list:
            loc = stay.get("location_code") or dest_code
            loc_code = (_code(loc) if isinstance(loc, dict) else str(loc or "").upper()) or (str(dest_code or "").upper() if dest_code else "")
            ci = stay.get("check_in")
            co = stay.get("check_out")
            n_guests = stay.get("number_of_guests", default_pax)
            tids = stay.get("traveler_ids") or []
            if not loc or not ci or not co:
                continue
            if prev_stay_loc and loc_code and str(loc_code).upper() != str(prev_stay_loc).upper():
                for seg in intermediate:
                    o = _code(seg.get("origin"))
                    d = _code(seg.get("destination"))
                    if str(o).upper() == str(prev_stay_loc).upper() and str(d).upper() == str(loc_code).upper():
                        dep = seg.get("depart_date")
                        pax = seg.get("passengers", default_pax)
                        seg_tids = seg.get("traveler_ids") or []
                        if o and d and dep:
                            steps.append(PlanStep(
                                step_id=len(steps),
                                title=f"{o} to {d} flight",
                                action="quote_flight",
                                inputs={
                                    "from_airport_code": o,
                                    "to_airport_code": d,
                                    "departure_date": dep,
                                    "leg": leg,
                                    "passengers": pax,
                                    "traveler_ids": seg_tids,
                                },
                                enter_guard="True",
                                success_criteria="len(result) > 0",
                                depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                                next_step=None,
                            ))
                            leg += 1
                        break
            prev_stay_loc = loc_code
            try:
                from datetime import datetime
                ci_dt = datetime.strptime(str(ci)[:10], "%Y-%m-%d")
                co_dt = datetime.strptime(str(co)[:10], "%Y-%m-%d")
                nights = max(1, (co_dt - ci_dt).days)
            except Exception:
                nights = 1
            n_guests_str = str(n_guests) if isinstance(n_guests, (int, float)) else n_guests
            steps.append(PlanStep(
                step_id=len(steps),
                title=f"{loc} hotel {nights} nights ({n_guests_str} guests)",
                action="quote_hotel",
                inputs={
                    "city": loc,
                    "area": None,
                    "check_in_date": ci,
                    "number_of_nights": str(nights),
                    "number_of_guests": n_guests_str,
                    "traveler_ids": tids,
                },
                enter_guard="True",
                success_criteria="len(result) > 0",
                depends_on=[len(steps) - 1] if steps else [],
                next_step=None,
            ))

        for seg in return_segs:
            o_code = _code(seg.get("origin"))
            d_code = _code(seg.get("destination"))
            dep = seg.get("depart_date")
            pax = seg.get("passengers", default_pax)
            tids = seg.get("traveler_ids") or []
            if o_code and d_code and dep:
                steps.append(PlanStep(
                    step_id=len(steps),
                    title=f"{o_code} to {d_code} flight",
                    action="quote_flight",
                    inputs={
                        "from_airport_code": o_code,
                        "to_airport_code": d_code,
                        "departure_date": dep,
                        "leg": leg,
                        "passengers": pax,
                        "traveler_ids": tids,
                    },
                    enter_guard="True",
                    success_criteria="len(result) > 0",
                    depends_on=[len(steps) - 1] if len(steps) > 0 else [],
                    next_step=None,
                ))
                leg += 1

        for i, step in enumerate(steps):
            step.step_id = i
            step.next_step = None if i == len(steps) - 1 else i + 1
            step.depends_on = [i - 1] if i > 0 else []

        return Plan(id=plan_id, steps=steps, meta={"strategy": "programmatic"})

    # Light plan generation: intent → plan (programmatic; no LLM)
    def compose_plan_light(self, intent: Dict[str, Any]) -> Plan:
        """Build plan from intent programmatically. Intent is the source of truth."""
        plan_id = uuid.uuid4().hex[:8]
        plan = self._build_plan_from_intent(intent, plan_id)
        return self._validate_and_patch_plan(plan)

    # Orchestrator - plan is reflection of intent (no examples needed; intent is explicit)
    def propose(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        function = 'propose'
        print('Generating plan from intent (light)...')
        print('Intent (preview):', json.dumps(intent_for_retrieval(intent))[:200])

        plan = self.compose_plan_light(intent)
        return {
            "function": "propose",
            "success": True,
            "input": intent,
            "output": {
                "intent": intent,
                "plan": asdict(plan),
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
        Returns a dictionary with keys: 'to_intent', 'compose_plan_light'
        """
        prompts = {
            'to_intent': '',
            'compose_plan_light': ''
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
                    elif key == 'compose_plan_light':
                        prompts['compose_plan_light'] = prompt_text
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

            
            # Step 1: Generate Intent
            
            retrieved = planner.retrieve(user_message, k_cases=4, k_facts=4, k_skills=6)
            intent = planner.to_intent(req, cases=retrieved.get('cases', []),
                                       facts=retrieved.get('facts', []), skills=retrieved.get('skills', []))
            if not intent:
                return {'success': False, 'function': function, 'input': payload, 'output': 'Intent could not be extracted'}

            self.AGU.mutate_workspace(
                {'request_intent': intent},
                public_user=public_user,
                workspace_id=workspace_id
            )
            print('Intent stored in workspace')
            
            
            # Step 2: Generate Plan

            from pes.handlers.propose_plan import ProposePlan
            propose_payload = {
                'portfolio': context.portfolio,
                'org': context.org,
                'case_group': context.case_group,
                'intent': intent,
                '_init': context.init,
                '_entity_type': entity_type,
                '_entity_id': entity_id,
                '_thread': thread,
            }
            response_1 = ProposePlan().run(propose_payload)
            results.append(response_1)
            if not response_1.get('success'):
                return {'success': False, 'output': results}

            canonical = results[-1]['output']
            return {'success': True, 'interface': 'plan', 'input': payload, 'output': canonical, 'stack': results}
            
        except Exception as e:
            print(f'Error during execution: {str(e)}')
            return {'success': False, 'function': function, 'input': payload, 'output': f'ERROR:@generate_plan/run: {str(e)}'}

