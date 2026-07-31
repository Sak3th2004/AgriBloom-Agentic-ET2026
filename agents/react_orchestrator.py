"""
ReAct orchestrator (Algorithm 4) — dynamic Thought -> Action -> Observation.

Rather than a fixed vision->knowledge->compliance->output order, the agent
decides its NEXT action each turn based on what it has already observed. We
implement the classic text-based ReAct loop (Yao et al. 2023) so it works with
ANY open-source LLM through the free ``genai_handler`` router — no vendor
tool-use API required.

Robustness first: the LLM only ever *chooses among currently-valid actions*
(computed deterministically in :mod:`graph.state`), its choice is validated, and
if the LLM is unavailable or replies with junk we fall back to the deterministic
policy. So the agent is dynamic when a model is present and still correct when
it isn't.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from agents.react_tools import tool_descriptions
from graph.state import (
    OUTPUT,
    deterministic_next,
    summarize_progress,
    valid_actions,
)

logger = logging.getLogger(__name__)

GenerateFn = Callable[[str], str]

# Routing decisions must stay snappy: bound the wait regardless of how slow
# the underlying provider fallback chain (NVIDIA -> Gemini -> Ollama) is.
DECISION_TIMEOUT_SECONDS = 8.0


def _call_with_budget(fn: GenerateFn, prompt: str, timeout_seconds: float) -> str:
    """Call ``fn(prompt)`` with a hard wall-clock budget; raises on timeout.

    Uses ``shutdown(wait=False)`` deliberately: if the provider chain is stuck
    well past our budget, we must return control immediately rather than block
    on the still-running background thread (that would defeat the timeout).
    The orphaned thread is daemonized by the interpreter at process exit.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn, prompt)
        return future.result(timeout=timeout_seconds)
    except FutureTimeout as e:
        raise TimeoutError(f"LLM decision exceeded {timeout_seconds}s budget") from e
    finally:
        pool.shutdown(wait=False)


_SYSTEM = (
    "You are the orchestrator of an agricultural advisory agent. Decide the "
    "SINGLE next action to take. Think briefly, then choose ONE action from the "
    "allowed list. Respond ONLY with JSON: "
    '{{"thought": "<one short sentence>", "action": "<action_name>"}}.\n\n'
    "Farmer request:\n{query}\n\n"
    "Progress so far:\n{progress}\n\n"
    "Allowed actions (choose exactly one):\n{tools}\n"
)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction from an LLM reply."""
    if not text:
        return None
    # Strip code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def parse_action(text: str, allowed: list[str]) -> Optional[str]:
    """Parse an action name from an LLM reply, restricted to ``allowed``.

    Accepts a JSON object with an ``action`` field, or a bare action word
    appearing in the text. Returns None if nothing valid is found.
    """
    obj = _extract_json(text)
    if obj is not None:
        action = str(obj.get("action", "")).strip().lower()
        if action in allowed:
            return action
    # Fallback: look for a bare allowed keyword in the raw text.
    lowered = (text or "").lower()
    for a in allowed:
        if re.search(rf"\b{re.escape(a)}\b", lowered):
            return a
    return None


def decide_next_action(
    state: dict,
    generate: Optional[GenerateFn] = None,
) -> str:
    """Choose the next action. LLM-driven when possible, deterministic otherwise.

    Args:
        state: the pipeline state dict.
        generate: optional text-completion function (injected for tests). When
            None, the free ``genai_handler`` router is used; if that's
            unavailable we fall back to the deterministic policy.
    """
    allowed = valid_actions(state)
    if not allowed:
        return OUTPUT
    if len(allowed) == 1:
        return allowed[0]

    gen = generate or _default_generate()
    if gen is None:
        return deterministic_next(state)

    prompt = _SYSTEM.format(
        query=state.get("user_text") or "(image only, no text)",
        progress=json.dumps(summarize_progress(state), ensure_ascii=False),
        tools=tool_descriptions(allowed),
    )
    try:
        reply = _call_with_budget(gen, prompt, timeout_seconds=DECISION_TIMEOUT_SECONDS)
    except Exception as e:
        # Routing only needs one of a few actions, so it must never sit through
        # the full provider fallback chain (NVIDIA -> Gemini -> Ollama, which
        # can take well over a minute when providers are rate-limited/down).
        # Bounding this call keeps the agent responsive; treatment/compliance
        # generation elsewhere keeps its own more patient timeouts.
        logger.warning("ReAct: LLM decision failed/slow (%s) — deterministic fallback", e)
        return deterministic_next(state)

    action = parse_action(reply, allowed)
    if action is None:
        logger.info("ReAct: unparseable decision — deterministic fallback")
        return deterministic_next(state)
    return action


def _default_generate() -> Optional[GenerateFn]:
    """Return the free multi-backend generator, or None if unavailable."""
    try:
        from utils import genai_handler

        if not genai_handler.is_genai_available():
            return None
        return lambda prompt: genai_handler._generate(prompt)
    except Exception:
        return None


def run_react_orchestrator(state: dict) -> dict:
    """LangGraph node: decide the next action and record it on the state."""
    from graph.state import iteration_guard

    guarded, exceeded = iteration_guard(state)
    if exceeded:
        logger.warning("ReAct: iteration guard tripped — forcing output")
        return {**guarded, "next_action": OUTPUT}
    action = decide_next_action(guarded)
    logger.info("ReAct decision: %s (valid=%s)", action, valid_actions(guarded))
    return {**guarded, "next_action": action}
