"""The model seam.

The loop never imports a language model. When Decide meets a situation it has
not seen, it can call an `explorer` - a callable (situation, actions) -> action -
to suggest what to try. This module gives you two ways to build one:

    offline(seed)         deterministic random choice, no dependencies
    from_completion(fn)   wrap any LLM: fn(prompt) -> text

Bring whatever model you like. `from_completion` turns a one-line completion
function into an explorer and stays safe by contract: if the model fails or
returns something that is not a legal action, it warns once and falls back to a
random legal action, so dropping a model into a policy never takes the loop down.

    from oodaa import Policy
    from oodaa.llm import from_completion

    def complete(prompt: str) -> str:
        return my_model.generate(prompt)   # any provider, your call

    policy = Policy(epsilon=0.15, explorer=from_completion(complete))
"""

from __future__ import annotations

import random
import sys
from typing import Callable

Explorer = Callable[[str, list], str]
Completion = Callable[[str], str]

# The instruction handed to a model when Decide explores an unfamiliar situation.
PROMPT = (
    "You are the Decide step of an OODAA control loop.\n"
    "Situation: {situation}\n"
    "Available actions: {actions}\n"
    "Reply with exactly one action from the list, and nothing else."
)


def offline(seed: int = 0) -> Explorer:
    """No network. Deterministic random choice - enough to bootstrap exploration."""
    rng = random.Random(seed)

    def explore(situation: str, actions: list) -> str:
        return rng.choice(actions)

    return explore


def from_completion(complete: Completion) -> Explorer:
    """Turn any text-completion function into an explorer.

    `complete` takes a prompt and returns the model's text. Best-effort: a failed
    call or an illegal answer degrades to a random legal action (warning once).
    """
    fallback = offline()
    warned = []

    def explore(situation: str, actions: list) -> str:
        try:
            text = complete(PROMPT.format(situation=situation, actions=actions)).strip()
            return text if text in actions else fallback(situation, actions)
        except Exception as exc:
            if not warned:
                warned.append(True)
                print(f"[oodaa] model explorer unavailable ({exc}); exploring randomly.",
                      file=sys.stderr)
            return fallback(situation, actions)

    return explore
