"""
Shared plan continuity, execution, and tool-call helpers for PES-style agents.

Callers supply ``AgentUtilities``, workspace resolution, and optional hooks so
forks (e.g. ``agent_quotes``) can keep different context wiring while sharing logic.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from renglo.agent.agent_utilities import AgentUtilities

from pes.handlers.execute_plan import ExecutePlan
from pes.handlers.plan_modify_classifier import classify_user_requests_plan_modify


ContinuityMode = Literal['resume', 'modify_intent', 'none']


@dataclass
class NoNextContinuityResolution:
    """
    Result when the client omits ``next`` (no explicit c_id).

    * ``modify_intent`` — user asked to edit an existing saved plan; do not synthesize
      a c_id or run ``continuity_router`` for execution resume; load plan into context
      and use the modifying-plan action.
    * ``resume`` — synthesize ``c_id`` and run the normal resume path.
    * ``none`` — fall through to ``continuity_router(None)`` / planner selection as before.
    """

    mode: ContinuityMode
    c_id: Optional[str] = None
    plan_id: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    plan_state: Optional[Dict[str, Any]] = None
    # Set when mode is ``modify_intent`` — why we treated the user message as a plan edit.
    plan_modify_signal_explanation: Optional[str] = None


def _plan_state_ready(plan_state: Any) -> bool:
    return isinstance(plan_state, dict) and 'steps' in plan_state


_TERMINAL_STEP_STATUSES = frozenset({'completed', 'success'})

PLAN_STEP_TERMINAL_STATUSES = frozenset({'completed', 'success', 'complete', 'done'})


def step_row_status_is_terminal(status: Any) -> bool:
    if status is None:
        return False
    s = str(status).strip().lower()
    return s in PLAN_STEP_TERMINAL_STATUSES


def is_plan_fully_terminal(workspace: Optional[Dict[str, Any]], plan_id: str) -> bool:
    if not workspace or not isinstance(workspace, dict) or not str(plan_id).strip():
        return False
    pid = str(plan_id).strip()
    plans = workspace.get('plan') or {}
    plan = plans.get(pid)
    if not isinstance(plan, dict):
        return False
    sm = workspace.get('state_machine') or {}
    plan_state = sm.get(pid)
    if not isinstance(plan_state, dict):
        return False
    step_rows = {str(s.get('step_id')): s for s in (plan_state.get('steps') or [])}
    for ps in plan.get('steps') or []:
        if not isinstance(ps, dict):
            continue
        sid = str(ps.get('step_id', ''))
        row = step_rows.get(sid)
        st = (row or {}).get('status')
        if not step_row_status_is_terminal(st):
            return False
    return True


def mark_plan_state_machine_closed(agu: AgentUtilities, plan_id: str) -> bool:
    if not agu or not str(plan_id).strip():
        return False
    pid = str(plan_id).strip()
    ws = agu.get_active_workspace()
    if not ws or not isinstance(ws, dict):
        return False
    if not is_plan_fully_terminal(ws, pid):
        return False
    sm = ws.get('state_machine') or {}
    ps = sm.get(pid)
    if not isinstance(ps, dict):
        return False
    out = deepcopy(ps)
    out['plan_id'] = out.get('plan_id') or pid
    out['status'] = 'completed'
    out['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    agu.mutate_workspace({'replace_state_machine': out})
    return True


def reconcile_terminal_plan_roots(agu: AgentUtilities) -> List[str]:
    if not agu:
        return []
    ws = agu.get_active_workspace()
    if not ws or not isinstance(ws, dict):
        return []
    updated: List[str] = []
    for pid in list((ws.get('plan') or {}).keys()):
        pid_s = str(pid).strip()
        if not pid_s:
            continue
        if not is_plan_fully_terminal(ws, pid_s):
            continue
        if mark_plan_state_machine_closed(agu, pid_s):
            updated.append(pid_s)
        ws = agu.get_active_workspace()
        if not ws or not isinstance(ws, dict):
            break
    return updated


def _step_signature_for_reconcile(step: Any) -> Tuple[Any, ...]:
    if not isinstance(step, dict):
        return ()
    inputs = step.get('inputs') or {}
    try:
        ins = json.dumps(inputs, sort_keys=True, default=str)
    except TypeError:
        ins = str(inputs)
    return (str(step.get('action') or ''), str(step.get('title') or ''), ins)


def first_diverging_step_index(old_steps: List[Any], new_steps: List[Any]) -> int:
    """
    First index where old and new plan steps differ (action/title/inputs), or first
    length mismatch, or len(new_steps) if prefixes are identical.
    """
    for i, pair in enumerate(zip(old_steps, new_steps)):
        o, n = pair
        if _step_signature_for_reconcile(o) != _step_signature_for_reconcile(n):
            return i
    if len(old_steps) != len(new_steps):
        return min(len(old_steps), len(new_steps))
    return len(new_steps)


def _fresh_step_row_for_reconcile(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'step_id': step.get('step_id'),
        'title': step.get('title'),
        'status': 'pending',
        'result': None,
        'error': None,
        'started_at': None,
        'finished_at': None,
    }


def reconcile_plan_state_in_place(
    old_plan: Dict[str, Any],
    new_plan: Dict[str, Any],
    old_state: Dict[str, Any],
    *,
    now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Option C: same ``plan_id`` with new ``steps`` — rebuild ``state_machine`` rows.

    Prefix steps that still match the old plan and were **terminal** (completed/success)
    keep their row; from the first diverging index onward, rows are reset to pending.
    Context ``step_execution`` is cleared whenever any step is reset.
    """
    now = now_iso or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    old_steps = old_plan.get('steps') or []
    new_steps = new_plan.get('steps') or []
    if not isinstance(old_steps, list):
        old_steps = []
    if not isinstance(new_steps, list):
        new_steps = []
    cut = first_diverging_step_index(old_steps, new_steps)
    old_rows = old_state.get('steps') if isinstance(old_state.get('steps'), list) else []
    plan_id = str(
        new_plan.get('id') or old_state.get('plan_id') or old_plan.get('id') or ''
    ).strip()

    new_rows: List[Dict[str, Any]] = []
    for i, ns in enumerate(new_steps):
        if not isinstance(ns, dict):
            continue
        if (
            i < cut
            and i < len(old_rows)
            and isinstance(old_rows[i], dict)
            and str(old_rows[i].get('status') or '') in _TERMINAL_STEP_STATUSES
        ):
            old_row = old_rows[i]
            new_rows.append(
                {
                    'step_id': ns.get('step_id'),
                    'title': ns.get('title'),
                    'status': old_row.get('status'),
                    'result': old_row.get('result'),
                    'error': old_row.get('error'),
                    'started_at': old_row.get('started_at'),
                    'finished_at': old_row.get('finished_at'),
                    'action_log': old_row.get('action_log') or [],
                }
            )
            continue
        new_rows.append(_fresh_step_row_for_reconcile(ns))

    old_ctx = old_state.get('context') if isinstance(old_state.get('context'), dict) else {}
    step_exec: Dict[str, Any] = {}
    if isinstance(old_ctx.get('step_execution'), dict):
        step_exec = dict(old_ctx.get('step_execution') or {})
    if cut < len(new_steps) or len(new_rows) != len(old_rows):
        step_exec = {}

    top_status = old_state.get('status') if isinstance(old_state, dict) else None
    if not isinstance(top_status, str) or not top_status.strip():
        top_status = 'pending'

    return {
        'plan_id': plan_id,
        'status': top_status,
        'created_at': (old_state.get('created_at') if isinstance(old_state, dict) else None)
        or now,
        'updated_at': now,
        'context': {'step_execution': step_exec},
        'steps': new_rows,
    }


def _plan_recency_tuple(workspace: Dict[str, Any], plan_id: str) -> Tuple[str, int]:
    """
    Ordering for max(): later plan wins.

    Uses state_machine updated_at / created_at (lexicographic works for ExecutePlan's
    ``%Y-%m-%dT%H:%M:%SZ``), then plan.meta timestamps if present, then bucket key order
    (Python dict insertion order: higher index = added later).
    """
    plans = workspace.get('plan') or {}
    keys = list(plans.keys())
    try:
        bucket_idx = keys.index(plan_id)
    except ValueError:
        bucket_idx = -1
    ts = ''
    st = (workspace.get('state_machine') or {}).get(plan_id)
    if isinstance(st, dict):
        ts = str(st.get('updated_at') or st.get('created_at') or '').strip()
    if not ts:
        doc = plans.get(plan_id)
        if isinstance(doc, dict):
            meta = doc.get('meta')
            if isinstance(meta, dict):
                ts = str(meta.get('updated_at') or meta.get('created_at') or '').strip()
    return (ts, bucket_idx)


def active_plan_bundle(
    workspace: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """
    Pick (plan_id, plan_doc, plan_state) for the workspace's active / resumable plan.
    Prefers a plan step that is awaiting/failed/running; otherwise any plan with state.
    When several plans qualify, prefers the most recently created/updated (see
    ``_plan_recency_tuple``).
    """
    plans = workspace.get('plan') or {}
    sm = workspace.get('state_machine') or {}
    if not plans:
        return None

    suspended = find_suspended_plan_step(workspace)
    if suspended:
        pid, _, _ = suspended
        if pid in plans:
            st = sm.get(pid)
            if _plan_state_ready(st):
                return pid, plans[pid], st

    ready: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for pid in plans.keys():
        st = sm.get(pid)
        if _plan_state_ready(st):
            ready.append((pid, plans[pid], st))
    if not ready:
        return None
    best = max(ready, key=lambda t: _plan_recency_tuple(workspace, t[0]))
    return best[0], best[1], best[2]


def _heuristic_plan_modify_substrings(m: str) -> bool:
    """
    Conservative, domain-neutral substrings: True only when wording clearly targets
    editing or replacing the *saved plan* (steps, workflow, or top-level plan scope),
    not routine continuation of the current step.

    PES is industry-agnostic — no travel- or market-specific phrases belong here.
    """
    hints = (
        'modify the plan',
        'change the plan',
        'change my plan',
        'update the plan',
        'revise the plan',
        'edit the plan',
        'adjust the plan',
        'alter the plan',
        'add a step',
        'remove a step',
        'reorder the steps',
        'reorder steps',
        'skip a step',
        'replace a step',
        'insert a step',
        'scratch the plan',
        'abandon the plan',
        'redo the plan',
        'replan',
        're-plan',
        'rewrite the plan',
        'replace the plan',
        'different plan',
        'start a different plan',
        'new plan instead',
        'change the scope',
        'narrow the scope',
        'expand the scope',
        'change requirements for the plan',
        'new requirements for the plan',
    )
    if any(h in m for h in hints):
        return True
    # Plan-level schedule/timeline shifts (generic — not tied to any one domain).
    timeline = (
        'change the timeline',
        'different timeline',
        'move the deadline',
        'extend the deadline',
        'change the deadline',
        'postpone the plan',
        'defer the plan',
        'reschedule the plan',
    )
    return any(h in m for h in timeline)


def _normalize_planner_intent(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not payload:
        return None
    v = payload.get('planner_intent') or payload.get('planner_mode')
    if v is None:
        return None
    s = str(v).strip().lower().replace('-', '_')
    if s in ('modify', 'modify_intent', 'modifying_plan', 'change_plan'):
        return 'modify'
    return None


def _evaluate_plan_modify_signal(
    user_message: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    if _normalize_planner_intent(payload) == 'modify':
        return True, (
            "Payload declares planner intent to change the saved plan "
            "(planner_intent / planner_mode)."
        )
    if not user_message or not str(user_message).strip():
        return False, 'No user text to classify for plan changes.'
    m = user_message.strip().lower()
    if "don't change the plan" in m or 'do not change the plan' in m:
        return False, 'User asked explicitly not to change the plan.'
    if _heuristic_plan_modify_substrings(m):
        return (
            True,
            'Matched conservative plan-change phrases (explicit intent to edit the saved plan).',
        )
    llm = classify_user_requests_plan_modify(
        str(user_message).strip(),
        config=config,
    )
    if llm is True:
        return True, (
            'Classifier reports a clear request to revise or replace the persisted plan '
            '(structure, steps, scope, or plan-level constraints).'
        )
    if llm is False:
        return False, (
            'Classifier: message is not a clear plan-change request; keeping current plan.'
        )
    return (
        False,
        'Classifier unavailable or inconclusive; default is to keep the current plan.',
    )


def explain_plan_modify_decision(
    user_message: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Human-readable reason for routing (or not) into the formal ``modify_intent`` side path.

    Used by callers (e.g. ``agent_quotes``) when logging why a yes/no gate was shown.
    Mirrors :func:`user_message_requests_plan_modify` logic.
    """
    return _evaluate_plan_modify_signal(user_message, payload, config=config)


def user_message_requests_plan_modify(
    user_message: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Whether this turn should use the formal "modify saved plan" path (no resume c_id).

    **Default is to keep the current plan** unless the caller explicitly flags modification in
    ``payload`` (``planner_intent`` / ``planner_mode``), a conservative phrase list matches
    clear edit intent, or the optional LLM classifier returns a confident True.

    When heuristics miss, an optional LLM classifier may run (``plan_modify_classifier``).
    If the classifier is disabled or inconclusive, the result stays False (no reroute).
    """
    decides, _ = _evaluate_plan_modify_signal(user_message, payload, config=config)
    return decides


def plan_modify_resolution_if_applicable(
    workspace: Optional[Dict[str, Any]],
    user_message: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[NoNextContinuityResolution]:
    """
    If a saved plan exists and the user message asks to change requirements or the plan,
    return a ``modify_intent`` resolution for the active plan bundle.

    Used both when ``next`` is omitted (see ``resolve_no_next_continuity``) and when a
    stale consent / resume ``next`` must be overridden (caller clears ``next``).
    """
    if not workspace or not isinstance(workspace, dict):
        return None
    if not (workspace.get('plan') or {}):
        return None
    local_probe, _ = classify_local_reply_against_pending_execution(
        workspace, user_message
    )
    if local_probe in ('match', 'ambiguous'):
        return None
    bundle = active_plan_bundle(workspace)
    if not bundle:
        return None
    if _execution_step_preempts_plan_modify(workspace, user_message):
        return None
    decides, rationale = _evaluate_plan_modify_signal(
        user_message, payload, config=config
    )
    if not decides:
        return None
    plan_id, plan_doc, plan_state = bundle
    return NoNextContinuityResolution(
        mode='modify_intent',
        plan_id=plan_id,
        plan=plan_doc,
        plan_state=plan_state,
        plan_modify_signal_explanation=rationale,
    )


def resolve_no_next_continuity(
    user_message: Optional[str],
    *,
    agu: AgentUtilities,
    payload: Optional[Dict[str, Any]] = None,
    get_workspace: Optional[Callable[[], Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> NoNextContinuityResolution:
    """
    When ``next`` was not sent, decide: resume execution (c_id), modify plan, or neither.

    When a step is ``awaiting`` with a nonce-bearing action-log entry for the current tool,
    we infer ``resume`` first (any natural-language continuation, not only short ``ok``),
    so picker-style lines are not mis-routed to ``modify_intent``. Short affirm/deny still
    use the dedicated consent path when applicable.
    """
    if not agu:
        return NoNextContinuityResolution(mode='none')

    if get_workspace is None:
        workspace = agu.get_active_workspace()
    else:
        workspace = get_workspace()

    if not workspace or not isinstance(workspace, dict):
        return NoNextContinuityResolution(mode='none')

    if not (workspace.get('plan') or {}):
        return NoNextContinuityResolution(mode='none')

    suspended = find_suspended_plan_step(workspace)
    if suspended:
        _plan_id_s, _step_id_s, step_row = suspended
        if _execution_step_preempts_plan_modify(workspace, user_message):
            c_id_resume = infer_continuity_id(
                user_message, agu=agu, get_workspace=lambda: workspace
            )
            if c_id_resume:
                return NoNextContinuityResolution(mode='resume', c_id=c_id_resume)
        if (
            isinstance(step_row, dict)
            and _step_row_awaits_tool_consent(step_row)
            and user_message_looks_like_tool_consent_reply(user_message)
        ):
            c_id_consent = infer_continuity_id(
                user_message, agu=agu, get_workspace=lambda: workspace
            )
            if c_id_consent:
                return NoNextContinuityResolution(mode='resume', c_id=c_id_consent)

    modify_res = plan_modify_resolution_if_applicable(
        workspace, user_message, payload, config=config
    )
    if modify_res:
        return modify_res

    c_id = infer_continuity_id(
        user_message, agu=agu, get_workspace=lambda: workspace
    )
    if c_id:
        return NoNextContinuityResolution(mode='resume', c_id=c_id)
    return NoNextContinuityResolution(mode='none')


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super().default(obj)


def action_log_last_nonce_entry(
    action_log: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not action_log:
        return None
    for entry in reversed(action_log):
        if entry.get('nonce') is not None:
            return entry
    return None


def _step_row_awaits_tool_consent(step_row: Dict[str, Any]) -> bool:
    """True when the specialist is waiting on binary tool consent (WAITING_HUMAN / status 3)."""
    if not isinstance(step_row, dict):
        return False
    entry = action_log_last_nonce_entry(step_row.get('action_log'))
    if not isinstance(entry, dict):
        return False
    if str(entry.get('type') or '').strip() != 'consent_rq':
        return False
    st = entry.get('status')
    return st == 3 or str(st).strip() == '3'


def user_message_looks_like_tool_consent_reply(user_message: Optional[str]) -> bool:
    """
    Short replies that should resume a pending tool consent, not ``modify_intent``.

    The plan-modify classifier can misfire on ``ok`` / ``yes``; those must still infer ``c_id``.
    """
    if not user_message or not str(user_message).strip():
        return False
    m = str(user_message).strip().lower()
    if len(m) > 160:
        return False
    if re.search(
        r'\b(change|changing|modify|modifying|instead|different|airport|flight|hotel|'
        r'nights?|add |remove |from [a-z]{3}\b|to [a-z]{3}\b)\b',
        m,
    ):
        return False
    if _heuristic_plan_modify_substrings(m):
        return False
    compact = re.sub(r'[\s,.!\'"]+', ' ', m).strip()
    affirm = (
        'yes',
        'y',
        'ok',
        'okay',
        'sure',
        'please do',
        'go ahead',
        'proceed',
        'confirm',
        'confirmed',
        'approved',
        'approve',
        'i approve',
        'sounds good',
        'looks good',
        'that works',
        'do it',
        'run it',
        'fine',
        'absolutely',
        'definitely',
        'yes please',
        'ok please',
        'please go ahead',
        'yes thanks',
        'ok thanks',
    )
    deny = (
        'no',
        'n',
        'cancel',
        'stop',
        "don't",
        'dont',
        'do not',
        'abort',
        'never mind',
        'nevermind',
        'no thanks',
    )
    if compact in affirm or compact in deny:
        return True
    if compact.startswith('yes ') and len(compact) < 90 and 'change' not in compact:
        return True
    return False


def classify_local_reply_against_pending_execution(
    workspace: Optional[Dict[str, Any]],
    user_message: Optional[str],
) -> Tuple[str, str]:
    """
    Local-first tri-state probe used by outer layers before plan-modify routing.

    Returns:
      - ``("match", reason)``: message likely responds to current specialist gate.
      - ``("ambiguous", reason)``: local response is plausible but not decisive.
      - ``("no_match", reason)``: not a local execution reply.
    """
    if not workspace or not isinstance(workspace, dict):
        return 'no_match', 'No workspace available for local execution probe.'

    found = find_most_advanced_execution_followup_hold(workspace)
    if not found:
        return 'no_match', 'No pending execution follow-up gate found.'

    _, _, step_row = found
    if not isinstance(step_row, dict):
        return 'no_match', 'Pending gate row is unavailable.'

    msg = str(user_message or '').strip()
    if not msg:
        return 'ambiguous', 'Pending gate exists but user message is empty.'
    m = msg.lower()

    if _heuristic_plan_modify_substrings(m):
        return 'no_match', 'Message has explicit plan-edit phrasing.'

    if _step_row_awaits_tool_consent(step_row):
        if user_message_looks_like_tool_consent_reply(msg):
            return 'match', 'Message matches pending consent yes/no reply.'
        if len(re.sub(r'[\s,.!\'"]+', ' ', m).strip().split()) <= 3:
            return 'ambiguous', 'Short reply while consent gate is pending.'
        return 'no_match', 'Consent gate pending but reply does not look like consent.'

    if _plan_step_expects_execution_user_followup(step_row):
        if re.search(r'\b(option|pick|choose|select)\b', m):
            return 'match', 'Selection verb detected for pending execution gate.'
        if re.search(r'\b(first|second|third|fourth|fifth|last)\b', m):
            return 'match', 'Ordinal selection detected for pending execution gate.'
        if re.search(r'\b(cheapest|earliest|latest|best)\b', m):
            return 'match', 'Comparative selection detected for pending execution gate.'
        if re.search(r'\b(this one|that one|book it|take it)\b', m):
            return 'match', 'Direct selection wording detected.'
        if re.search(r'\b#?\d+\b', m):
            return 'match', 'Numeric option reference detected.'
        if m in {'yes', 'ok', 'okay', 'sure'}:
            return 'ambiguous', 'Ack reply while selection gate is pending.'
        if len(m.split()) <= 3:
            return 'ambiguous', 'Short reply while selection gate is pending.'
        return 'no_match', 'Pending execution gate exists but message did not bind locally.'

    return 'no_match', 'Pending row does not expose a local follow-up gate.'


def user_message_suggests_plan_continue(user_message: Optional[str]) -> bool:
    """
    True when the user is asking to resume or advance an existing saved plan.

    Used with ``find_pending_plan_step``: only then do we synthesize a c_id for the
    next ``pending`` step (avoid jumping plan execution on unrelated chit-chat).
    """
    if not user_message or not str(user_message).strip():
        return False
    m = user_message.lower()
    hints = (
        'continue with the plan',
        'continue the plan',
        'resume the plan',
        'pick up the plan',
        'keep going with the plan',
        'proceed with the plan',
        'go ahead with the plan',
        'next step of the plan',
        'run the next step',
        'finish the plan',
        'complete the plan',
        'execute the plan',
        'follow the plan',
    )
    return any(h in m for h in hints)


def user_message_suggests_fresh_search(user_message: Optional[str]) -> bool:
    if not user_message or not str(user_message).strip():
        return False
    m = user_message.lower()
    hints = (
        'search again',
        'redo search',
        'new search',
        'run search again',
        're-run search',
        'rerun search',
        'different dates',
        'change dates',
        'change the dates',
        'start over',
        'from scratch',
        'another search',
    )
    if "don't search" in m or 'do not search' in m:
        return False
    return any(h in m for h in hints)


def find_suspended_plan_step(
    workspace: Dict[str, Any],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Best suspended step for inferring a c_id: min status rank (awaiting < failed < running),
    then min step_id (string order, same as before), then newest plan if still tied.
    Only plans that already have matching state_machine rows are considered.
    """
    plans = workspace.get('plan') or {}
    sm = workspace.get('state_machine') or {}
    candidates: List[Tuple[int, str, str, Dict[str, Any]]] = []
    rank_map = {'awaiting': 0, 'failed': 1, 'running': 2}
    for plan_id in plans.keys():
        plan_state = sm.get(plan_id)
        if not isinstance(plan_state, dict):
            continue
        for step in plan_state.get('steps') or []:
            status = step.get('status')
            if status not in rank_map:
                continue
            sid = str(step.get('step_id', ''))
            candidates.append((rank_map[status], plan_id, sid, step))
    if not candidates:
        return None
    min_rank = min(c[0] for c in candidates)
    pool = [c for c in candidates if c[0] == min_rank]
    min_sid = min(c[2] for c in pool)
    pool = [c for c in pool if c[2] == min_sid]
    best = max(pool, key=lambda c: _plan_recency_tuple(workspace, c[1]))
    _, plan_id, sid, step = best
    return plan_id, sid, step


def _nonce_entry_indicates_execution_user_followup(entry: Any) -> bool:
    """
    Latest nonce-bearing action-log row implies the user should reply so execution can resume
    (consent/decision/tool completion handoff—not a saved-plan rewrite).
    """
    if not isinstance(entry, dict) or entry.get('nonce') is None:
        return False
    t = str(entry.get('type') or '').strip().lower()
    if t in ('consent_rq', 'decision_rq', 'tool_ok'):
        return True
    if entry.get('yield_for_user') is True:
        return True
    tool = entry.get('tool')
    return bool(tool and str(tool).strip() not in ('', '*'))


def _plan_step_expects_execution_user_followup(step_row: Dict[str, Any]) -> bool:
    """True when this step row likely awaits user follow-up for execution recovery/continuation."""
    if not isinstance(step_row, dict):
        return False
    if str(step_row.get('status') or '').strip().lower() not in (
        'awaiting',
        'running',
        'failed',
    ):
        return False
    entry = action_log_last_nonce_entry(step_row.get('action_log'))
    return _nonce_entry_indicates_execution_user_followup(entry)


def _execution_step_preempts_plan_modify(
    workspace: Dict[str, Any],
    user_message: Optional[str],
) -> bool:
    """
    True when any plan step is ``awaiting``/``running`` with a nonce handoff expecting the user
    to continue the **current tool/workflow** response—saved-plan rewrite routing must not run unless
    the message explicitly matches ``_heuristic_plan_modify_substrings``.

    We scan **all** qualifying steps instead of relying on :func:`find_suspended_plan_step` alone,
    because multiple steps may still be marked ``awaiting``; picking only the smallest ``step_id``
    can hide the flight/hotel-picker step whose action_log carries the resume nonce.
    """
    m = str(user_message or '').strip().lower()
    if _heuristic_plan_modify_substrings(m):
        return False
    plans = workspace.get('plan') or {}
    sm = workspace.get('state_machine') or {}
    for plan_id in plans.keys():
        plan_state = sm.get(plan_id)
        if not isinstance(plan_state, dict):
            continue
        for row in plan_state.get('steps') or []:
            if isinstance(row, dict) and _plan_step_expects_execution_user_followup(row):
                return True
    return False


def _advancement_key_for_execution_followup_hold(
    workspace: Dict[str, Any],
    *,
    plan_id: str,
    step_id_raw: Any,
) -> Tuple[int, Tuple[str, int], str]:
    """
    Larger means a “later” resume anchor when multiple steps qualify for follow-up wording.

    Prefer larger numeric ``step_id`` when possible; break ties via plan recency.
    """
    sid = str(step_id_raw if step_id_raw is not None else '').strip()
    try:
        n = int(sid)
    except ValueError:
        n = -1
    rec = _plan_recency_tuple(workspace, plan_id)
    return (n, rec, sid)


def find_most_advanced_execution_followup_hold(
    workspace: Dict[str, Any],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Among steps awaiting execution follow-up (nonce-bearing handoffs), prefer the furthest progressed.

    Mirrors :func:`_execution_step_preempts_plan_modify` intent so ``infer_continuity_id`` can build
    a ``c_id`` that matches picker UX (e.g. hotel/flight rows) rather than an older stalled step id.
    """
    plans = workspace.get('plan') or {}
    sm = workspace.get('state_machine') or {}
    picked: Optional[Tuple[str, str, Dict[str, Any]]] = None
    best_k: Optional[Tuple[int, Tuple[str, int], str]] = None

    for plan_id in plans.keys():
        plan_state = sm.get(plan_id)
        if not isinstance(plan_state, dict):
            continue
        for row in plan_state.get('steps') or []:
            if not isinstance(row, dict):
                continue
            if not _plan_step_expects_execution_user_followup(row):
                continue
            rk = _advancement_key_for_execution_followup_hold(
                workspace,
                plan_id=str(plan_id),
                step_id_raw=row.get('step_id'),
            )
            if best_k is None or rk > best_k:
                best_k = rk
                picked = (
                    str(plan_id),
                    str(
                        row.get('step_id')
                        if row.get('step_id') is not None
                        else ''
                    ).strip(),
                    row,
                )
    return picked


def find_pending_plan_step(
    workspace: Dict[str, Any],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    First step in plan order that is still ``pending`` in the state machine and whose
    ``depends_on`` steps are terminal (``completed`` / ``success``).

    Used when there is no awaiting/failed/running step — e.g. user says \"continue\"
    after prior steps finished but the executor never started the next one.
    """
    plans = workspace.get('plan') or {}
    sm = workspace.get('state_machine') or {}
    terminal = frozenset({'completed', 'success'})
    per_plan: List[Tuple[str, str, Dict[str, Any]]] = []
    for plan_id, plan_doc in plans.items():
        if not isinstance(plan_doc, dict):
            continue
        plan_state = sm.get(plan_id)
        if not isinstance(plan_state, dict):
            continue
        steps = plan_doc.get('steps') or []
        step_states_by_id = {
            str(s.get('step_id')): s for s in (plan_state.get('steps') or [])
        }
        for step in steps:
            if not isinstance(step, dict):
                continue
            sid = str(step.get('step_id', ''))
            row = step_states_by_id.get(sid)
            if not row or row.get('status') != 'pending':
                continue
            deps = step.get('depends_on') or []
            deps_ok = True
            for d in deps:
                dep_status = (step_states_by_id.get(str(d)) or {}).get('status')
                if dep_status not in terminal:
                    deps_ok = False
                    break
            if deps_ok:
                per_plan.append((plan_id, sid, row))
                break
    if not per_plan:
        return None
    return max(per_plan, key=lambda t: _plan_recency_tuple(workspace, t[0]))


def find_resume_target_step(
    workspace: Dict[str, Any],
    user_message: Optional[str] = None,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Prefer the most progressed execution follow-up anchor when several steps qualify, then fall
    back to generic suspension discovery, otherwise the dependency-ready ``pending`` step pattern.
    """
    follow = find_most_advanced_execution_followup_hold(workspace)
    if follow:
        return follow
    suspended = find_suspended_plan_step(workspace)
    if suspended:
        return suspended
    if user_message_suggests_plan_continue(user_message):
        return find_pending_plan_step(workspace)
    return None


def infer_continuity_id(
    user_message: Optional[str],
    *,
    agu: AgentUtilities,
    get_workspace: Optional[Callable[[], Any]] = None,
) -> Optional[str]:
    """
    Build a continuity id when the client omits ``next``.

    If ``get_workspace`` is omitted, uses ``agu.get_active_workspace()``.

    Resumes suspended steps (awaiting/failed/running) or, if none, starts the next
    ``pending`` step whose dependencies are completed.
    """
    if not agu:
        return None

    if get_workspace is None:
        workspace = agu.get_active_workspace()
    else:
        workspace = get_workspace()

    if not workspace or not isinstance(workspace, dict):
        return None

    if not (workspace.get('plan') or {}):
        return None

    found = find_resume_target_step(workspace, user_message)
    if not found:
        return None
    plan_id, plan_step, step = found
    if is_plan_fully_terminal(workspace, plan_id):
        return None

    if user_message_suggests_fresh_search(user_message):
        return f'irn:c_id:{plan_id}:{plan_step}:'

    entry = action_log_last_nonce_entry(step.get('action_log'))
    if not entry:
        return f'irn:c_id:{plan_id}:{plan_step}:'

    tool = entry.get('tool')
    if not tool:
        return f'irn:c_id:{plan_id}:{plan_step}:'

    tool_step = entry.get('status')
    if tool_step is None or tool_step == '':
        return f'irn:c_id:{plan_id}:{plan_step}:'

    nonce = entry.get('nonce')
    if nonce is None:
        return f'irn:c_id:{plan_id}:{plan_step}:'

    print(
        "infer_continuity_id(): inferred resume nonce "
        f"plan_id={plan_id}, plan_step={plan_step}, tool={tool}, status={tool_step}, nonce={nonce}"
    )

    return f'irn:c_id:{plan_id}:{plan_step}:{tool}:{tool_step}:{nonce}'


def continuity_router(
    c_id: Optional[str],
    *,
    agu: AgentUtilities,
    get_workspace: Callable[[], Any],
    update_plan_context: Optional[Callable[[Any, Any], None]] = None,
    silent_no_c_id: bool = False,
    special_c_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Route continuity requests. ``get_workspace`` must return a workspace dict
    (or falsy). ``update_plan_context(plan, plan_state)`` runs when a plan is resolved.

    ``special_c_id`` maps exact ``c_id`` strings to router ``output`` bodies
    (must include ``next_action`` and ``message``).
    """
    function = 'continuity_router'
    try:

        if not c_id:
            if not silent_no_c_id:
                agu.print_chat('No c_id provided...', 'transient')
            return {
                'success': True,
                'function': function,
                'input': c_id,
                'output': {
                    'next_action': 'to_be_determined',
                    'message': 'No continuity id, Creating new plan',
                },
            }

        if special_c_id and c_id in special_c_id:
            body = dict(special_c_id[c_id])
            return {
                'success': True,
                'function': function,
                'input': c_id,
                'output': body,
            }

        print(f'continuity_router(): incoming c_id={c_id!r}')
        parts = c_id.split(':')

        if parts[0] != 'irn' or parts[1] != 'c_id':
            pr = 'The continuity id is not valid.'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {'next_action': 'c_id_error', 'message': pr},
            }

        if not parts[2]:
            pr = f'Error: There is no plan_id in the c_id: ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {'next_action': 'c_id_error', 'message': pr},
            }

        plan_id = parts[2]
        workspace = get_workspace()
        if not workspace or not isinstance(workspace, dict):
            pr = 'No workspace available for continuity routing.'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {'next_action': 'c_id_error', 'message': pr},
            }

        plan_bucket = workspace.get('plan') or {}
        if plan_id not in plan_bucket:
            pr = 'No valid plan, generating new one.'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {'next_action': 'c_id_error', 'message': pr},
            }

        plan = plan_bucket[plan_id]
        sm = workspace.get('state_machine') or {}
        plan_state = sm.get(plan_id)
        if not isinstance(plan_state, dict):
            pr = 'Plan state missing for continuity id.'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {'next_action': 'c_id_error', 'message': pr},
            }

        if is_plan_fully_terminal(workspace, plan_id):
            pr = (
                'This plan is already fully executed; continuity for this plan is closed. '
                'Return control to triage or start a new request.'
            )
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'plan_execution_complete',
                    'plan_id': plan_id,
                    'message': pr,
                },
            }

        if update_plan_context:
            update_plan_context(plan, plan_state)

        if not parts[3]:
            pr = f'Starting from the first step... ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'initiate_plan',
                    'plan_id': plan_id,
                    'plan_step': 0,
                    'message': pr,
                },
            }

        plan_step = parts[3]
        step_exists = False
        for p_step in plan.get('steps', []):
            step_id = p_step.get('step_id')
            if step_id is not None and plan_step == str(step_id):
                step_exists = True
                break

        if not step_exists:
            pr = 'There is a plan, the step is not valid, starting from the first step'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'initiate_plan',
                    'plan_id': plan_id,
                    'plan_step': 0,
                    'message': pr,
                },
            }

        step_state_row = None
        for step in plan_state.get('steps') or []:
            if str(step.get('step_id')) == plan_step:
                step_state_row = step
                break
        resume_entry = (
            action_log_last_nonce_entry(
                step_state_row.get('action_log') if isinstance(step_state_row, dict) else None
            )
            if isinstance(step_state_row, dict)
            else None
        )
        if isinstance(resume_entry, dict):
            print(
                'continuity_router(): step action_log resume candidate '
                f"plan_id={plan_id}, plan_step={plan_step}, tool={resume_entry.get('tool')}, "
                f"status={resume_entry.get('status')}, nonce={resume_entry.get('nonce')}"
            )
        else:
            print(
                'continuity_router(): no step action_log resume candidate '
                f"plan_id={plan_id}, plan_step={plan_step}"
            )

        if not parts[4]:
            # If caller omitted the action/tool tail, prefer resuming the last pending
            # consent/tool handshake for this step instead of restarting from initiate_action.
            if (
                isinstance(resume_entry, dict)
                and resume_entry.get('tool')
                and resume_entry.get('status') not in (None, '')
                and resume_entry.get('nonce') is not None
            ):
                action_step = str(resume_entry.get('tool'))
                tool_step = resume_entry.get('status')
                pr = f'Resuming tool from step action_log... ({c_id})'
                agu.print_chat(pr, 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {
                        'next_action': 'resume_tool',
                        'plan_id': plan_id,
                        'plan_step': plan_step,
                        'action_step': action_step,
                        'tool_step': tool_step,
                        'message': pr,
                    },
                }
            pr = f'Starting the new action execution... ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'initiate_action',
                    'plan_id': plan_id,
                    'plan_step': plan_step,
                    'action_step': 0,
                    'message': pr,
                },
            }

        action_step = parts[4]
        if (
            isinstance(resume_entry, dict)
            and resume_entry.get('tool')
            and resume_entry.get('status') not in (None, '')
            and resume_entry.get('nonce') is not None
            and str(action_step) in ('0', '')
        ):
            pr = f'Resuming pending consent/tool from step action_log... ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'resume_tool',
                    'plan_id': plan_id,
                    'plan_step': plan_step,
                    'action_step': str(resume_entry.get('tool')),
                    'tool_step': resume_entry.get('status'),
                    'message': pr,
                },
            }

        if not parts[5]:
            if (
                isinstance(resume_entry, dict)
                and str(resume_entry.get('tool') or '') == str(action_step)
                and resume_entry.get('status') not in (None, '')
                and resume_entry.get('nonce') is not None
            ):
                pr = f'Resuming tool from step action_log... ({c_id})'
                agu.print_chat(pr, 'transient')
                return {
                    'success': True,
                    'input': c_id,
                    'output': {
                        'next_action': 'resume_tool',
                        'plan_id': plan_id,
                        'plan_step': plan_step,
                        'action_step': action_step,
                        'tool_step': resume_entry.get('status'),
                        'message': pr,
                    },
                }
            pr = f'Starting tool... ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'initiate_tool',
                    'plan_id': plan_id,
                    'plan_step': plan_step,
                    'action_step': action_step,
                    'tool_step': 0,
                    'message': pr,
                },
            }

        tool_step = parts[5]
        if (
            isinstance(resume_entry, dict)
            and str(resume_entry.get('tool') or '') == str(action_step)
            and resume_entry.get('status') not in (None, '')
            and resume_entry.get('nonce') is not None
            and str(tool_step) in ('0', '')
        ):
            pr = f'Resuming pending consent/tool from step action_log... ({c_id})'
            agu.print_chat(pr, 'transient')
            return {
                'success': True,
                'input': c_id,
                'output': {
                    'next_action': 'resume_tool',
                    'plan_id': plan_id,
                    'plan_step': plan_step,
                    'action_step': str(resume_entry.get('tool')),
                    'tool_step': resume_entry.get('status'),
                    'message': pr,
                },
            }
        print(f'Original tool_step: {tool_step}')
        if len(parts) > 6:
            print(f'Nonce: {parts[6]}', type(parts[6]))

        if len(parts) <= 6 or not parts[6]:
            print('No nonce, making tool_step = 0')
            tool_step = 0
        else:
            if step_state_row is None:
                print(
                    f'Step with step_id {plan_step} not found in plan_state, resetting tool_step = 0'
                )
                tool_step = 0
            else:
                action_log = step_state_row.get('action_log', [])
                nonce = None
                for entry in reversed(action_log):
                    if 'nonce' in entry:
                        nonce = entry['nonce']
                        break

                if nonce is None:
                    print('No nonce found in action_log, resetting tool_step = 0')
                    tool_step = 0
                else:
                    print(type(nonce))
                    if int(nonce) != int(parts[6]):
                        print('Invalid nonce, resetting tool_step = 0')
                        tool_step = 0
                    else:
                        print(f'Nonce has been verified, tool_step = {tool_step}')

        next_action = 'resume_tool'
        pr = f'There is a plan and a step and an action and a tool resume point... ({c_id})'
        agu.print_chat(pr, 'transient')

        return {
            'success': True,
            'input': c_id,
            'output': {
                'next_action': next_action,
                'plan_id': plan_id,
                'plan_step': plan_step,
                'action_step': action_step,
                'tool_step': tool_step,
                'message': pr,
            },
        }

    except Exception as e:
        pr = f'🤖❌ @continuity_router:{e}'
        agu.print_chat(pr, 'error')
        return {
            'success': True,
            'function': function,
            'input': c_id,
            'output': {'next_action': 'to_be_determined', 'message': pr},
        }


def run_plan_action_tool(
    plan_id: str,
    plan_step: str,
    action_step: str,
    tool_step: str,
    *,
    agu: AgentUtilities,
    execute_plan_cls: type = ExecutePlan,
    executor_tool_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    function = inspect.currentframe().f_code.co_name
    print('Running:', function)
    payload = {
        'plan_id': plan_id,
        'plan_step': plan_step,
        'action_step': action_step,
        'tool_step': tool_step,
    }
    try:
        executor = execute_plan_cls(agu, executor_tool_settings=executor_tool_settings)
        execution_result = executor.run(payload)
        if not execution_result['success']:
            return {
                'success': False,
                'function': function,
                'input': payload,
                'output': execution_result['output'],
            }
        print(json.dumps(execution_result, indent=2, cls=DecimalEncoder))
        return {
            'success': True,
            'function': function,
            'input': payload,
            'output': execution_result,
        }
    except Exception as e:
        pr = f'🤖❌ @rpat:{e}'
        agu.print_chat(pr, 'error')
        return {'success': False, 'function': function, 'input': payload, 'output': pr}


def execute_plan(
    next_action: str,
    plan_id: str,
    plan_step: str,
    action_step: str,
    tool_step: str,
    *,
    agu: AgentUtilities,
    execute_plan_cls: type = ExecutePlan,
    log_style: str = 'transient',
    log_label: str = 'execute_plan',
    executor_tool_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pr = f'Calling the executor for plan:{plan_id}, plan_step:{plan_step}, action_step:{action_step}, tool_step:{tool_step}'
    if log_style and log_style != 'none':
        agu.print_chat(pr, log_style)

    rpat_kw = {
        'agu': agu,
        'execute_plan_cls': execute_plan_cls,
        'executor_tool_settings': executor_tool_settings,
    }

    if next_action == 'initiate_plan':
        response = run_plan_action_tool(plan_id, 0, 0, 0, **rpat_kw)
    elif next_action == 'initiate_action':
        response = run_plan_action_tool(plan_id, plan_step, 0, 0, **rpat_kw)
    elif next_action == 'initiate_tool':
        response = run_plan_action_tool(
            plan_id, plan_step, action_step, 0, **rpat_kw
        )
    elif next_action == 'resume_tool':
        response = run_plan_action_tool(
            plan_id,
            plan_step,
            action_step,
            tool_step,
            **rpat_kw,
        )
    else:
        return {
            'success': False,
            'function': log_label,
            'input': f'{plan_id}:{plan_step}:{action_step}:{tool_step}',
            'output': f'Unknown next_action: {next_action}',
        }

    return {
        'success': True,
        'function': log_label,
        'input': f'{plan_id}:{plan_step}:{action_step}:{tool_step}',
        'output': response,
    }


def _extract_interface(response: Dict[str, Any]) -> Optional[Any]:
    if 'interface' in response:
        return response['interface']
    out = response.get('output')
    if isinstance(out, dict) and 'interface' in out:
        return out['interface']
    if (
        isinstance(out, list)
        and len(out) > 0
        and isinstance(out[0], dict)
        and 'interface' in out[0]
    ):
        return out[0]['interface']
    return None


def act(
    execution_request: Dict[str, Any],
    *,
    agu: AgentUtilities,
    schc: Any,
    list_tools: List[Dict[str, Any]],
    portfolio: str,
    org: str,
    entity_type: str,
    entity_id: str,
    thread: str,
    context_update: Optional[Callable[..., None]] = None,
    public_user: Optional[str] = None,
    workspace_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = 'act'
    tool_name: Optional[str] = None

    list_handlers: Dict[str, str] = {}
    list_inits: Dict[str, Any] = {}
    for t in list_tools:
        list_handlers[t.get('key', '')] = t.get('handler', '')
        init_value = t.get('init', {})
        if isinstance(init_value, str):
            try:
                init_value = json.loads(init_value)
            except (json.JSONDecodeError, ValueError):
                init_value = {}
        list_inits[t.get('key', '')] = init_value if isinstance(init_value, dict) else {}

    if context_update:
        context_update(list_handlers=list_handlers)

    try:
        tool_name = execution_request['tool_calls'][0]['function']['name']
        params = execution_request['tool_calls'][0]['function']['arguments']
        if isinstance(params, str):
            params = json.loads(params)
        tid = execution_request['tool_calls'][0]['id']

        print(f'tid:{tid}')

        if not tool_name:
            raise ValueError('❌ No tool name provided in tool selection')

        print(f'Selected tool: {tool_name}')
        agu.print_chat(
            f'Calling tool {tool_name} with parameters {params} ', 'transient'
        )
        print(f'Parameters: {params}')

        if tool_name not in list_handlers:
            error_msg = f"❌ No handler found for tool '{tool_name}'"
            print(error_msg)
            agu.print_chat(error_msg, 'error')
            raise ValueError(error_msg)

        if list_handlers[tool_name] == '':
            error_msg = '❌ Handler is empty'
            print(error_msg)
            agu.print_chat(error_msg, 'error')
            raise ValueError(error_msg)

        handler_init: Dict[str, Any] = {}
        if not isinstance(list_inits[tool_name], str) and isinstance(
            list_inits[tool_name], dict
        ):
            handler_init = list_inits[tool_name]

        handler_route = list_handlers[tool_name]
        parts = handler_route.split('/')
        if len(parts) != 2:
            error_msg = f'❌ {tool_name} is not a valid tool.'
            print(error_msg)
            agu.print_chat(error_msg, 'error')
            raise ValueError(error_msg)

        params['_portfolio'] = portfolio
        params['_org'] = org
        params['_entity_type'] = entity_type
        params['_entity_id'] = entity_id
        params['_thread'] = thread
        params['_init'] = handler_init

        if extra and isinstance(extra, dict):
            merge_extra = dict(extra)
            if tool_name == 'commit_plan':
                # ``commit_plan`` must use the tool JSON (draft) or its own cache lookup — not
                # ReAct ``extra``, which carries the *already committed* workspace plan/intent and
                # would overwrite the new plan the model (or forced tool) just supplied.
                merge_extra.pop('plan', None)
                merge_extra.pop('intent', None)
                merge_extra.pop('state_machine', None)
            params.update(merge_extra)

        print(f'Calling {handler_route} ')

        response = schc.handler_call(portfolio, org, parts[0], parts[1], params)

        print(f'Handler response:{response}')

        if not response['success']:
            return {'success': False, 'action': action, 'input': params, 'output': response}

        clean_output = response['output']
        clean_output_str = json.dumps(clean_output, cls=DecimalEncoder)

        interface = _extract_interface(response)

        tool_out = {
            'role': 'tool',
            'tool_call_id': f'{tid}',
            'content': clean_output_str,
            'tool_calls': False,
        }

        save_kw: Dict[str, Any] = {}
        if connection_id:
            save_kw['connection_id'] = connection_id
        if interface:
            agu.save_chat(tool_out, interface=interface, **save_kw)
        else:
            agu.save_chat(tool_out, **save_kw)

        if context_update:
            context_update(execute_intention_results=tool_out)

        index = f'irn:tool_rs:{handler_route}'
        tool_input = execution_request['tool_calls'][0]['function']['arguments']
        tool_input_obj = (
            json.loads(tool_input) if isinstance(tool_input, str) else tool_input
        )
        value = {'input': tool_input_obj, 'output': clean_output}
        mutate_kw: Dict[str, Any] = {}
        if public_user is not None:
            mutate_kw['public_user'] = public_user
        if workspace_id:
            mutate_kw['workspace_id'] = workspace_id
        agu.mutate_workspace({'cache': {index: value}}, **mutate_kw)

        print('✅ Tool execution complete.')

        return {'success': True, 'action': action, 'input': execution_request, 'output': tool_out}

    except Exception as e:
        tname = tool_name if tool_name else 'unknown'
        error_msg = f'❌ Execute Intention failed. @act trying to run tool:{tname!r}: {str(e)}'
        agu.print_chat(error_msg, 'error')
        print(error_msg)
        if context_update:
            context_update(execute_intention_error=error_msg)
        error_result = {
            'success': False,
            'action': action,
            'input': execution_request,
            'output': str(e),
        }
        if context_update:
            context_update(execute_intention_results=error_result)
        return error_result
