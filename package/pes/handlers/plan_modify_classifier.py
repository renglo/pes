"""
LLM fallback when heuristic substring matching misses plan/requirement-change intent.

Config (via ``load_config()`` or caller-supplied dict):

- ``OPENAI_API_KEY`` — required for classification.
- ``PLAN_MODIFY_INTENT_CLASSIFIER_ENABLED`` — optional; default True when API key is set.
  Set to false / 0 / no / off to disable (heuristics only).
- ``PLAN_MODIFY_INTENT_CLASSIFIER_MODEL`` — optional; default ``gpt-4o-mini``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from renglo.common import load_config

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


def plan_modify_classifier_enabled(config: Optional[Dict[str, Any]]) -> bool:
    if not config:
        return False
    key = str(config.get('OPENAI_API_KEY') or '').strip()
    if not key:
        return False
    v = config.get('PLAN_MODIFY_INTENT_CLASSIFIER_ENABLED')
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ('1', 'true', 'yes', 'y', 'on')


def _parse_plan_modify_flag(text: str) -> Optional[bool]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'plan_modify_intent' in obj:
            return bool(obj['plan_modify_intent'])
    except json.JSONDecodeError:
        pass
    m = re.search(
        r'"plan_modify_intent"\s*:\s*(true|false)',
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).lower() == 'true'
    return None


def classify_user_requests_plan_modify(
    user_message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """
    Classify whether the user clearly intends to revise the persisted execution plan
    (steps, scope, or plan-level constraints). Domain-agnostic; biased toward **no change**
    when the message is ambiguous.

    Returns
    -------
    True / False when the model responds parseably.
    None when classification is disabled, OpenAI is unavailable, or the call fails.
    """
    cfg = config if config is not None else load_config()
    if not plan_modify_classifier_enabled(cfg):
        return None
    if OpenAI is None:
        return None

    msg = (user_message or '').strip()
    if not msg:
        return None

    model = str(
        cfg.get('PLAN_MODIFY_INTENT_CLASSIFIER_MODEL') or 'gpt-4o-mini'
    ).strip()

    system = (
        'You help an assistant that executes a saved multi-step plan. The user message may refer '
        'to any domain (there is no specific industry). '
        'Reply with JSON only: {"plan_modify_intent": true} or {"plan_modify_intent": false}. '
        'Set plan_modify_intent to TRUE only when the user CLEARLY wants to revise, replace, '
        'rescope, or restructure the persisted plan itself—e.g. add/remove/reorder steps, replace '
        'the plan, change top-level requirements embodied in the plan, or shift deadlines/timeline '
        'at the plan level. '
        'Set FALSE by default when uncertain. '
        'FALSE includes: short confirmations or denials, choosing how to proceed within the current '
        'step, operational details that do not imply rewriting the plan, status questions, '
        'clarifications, greetings, or vague language that could go either way. '
        'If context is provided, use it: if the user is clearly responding to a pending confirmation/'
        'selection gate for the current step, prefer false unless the message explicitly asks to '
        'rewrite plan-level constraints.'
    )
    user_block = f'User message:\n"""{msg[:4000]}"""'
    if isinstance(context, dict) and context:
        try:
            cjson = json.dumps(context, ensure_ascii=True, sort_keys=True, default=str)
        except Exception:
            cjson = str(context)
        user_block += f'\n\nContext (JSON):\n{cjson[:6000]}'

    try:
        client = OpenAI(api_key=str(cfg.get('OPENAI_API_KEY') or '').strip())
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_block},
            ],
        )
        raw = (resp.choices[0].message.content or '').strip()
        parsed = _parse_plan_modify_flag(raw)
        return parsed
    except Exception:
        return None
