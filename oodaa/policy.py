"""Policy - the Decide step.

Given the current observation and what memory knows, pick an action. The default
is epsilon-greedy: usually take the best known action, occasionally explore.
Two cases force exploration regardless of epsilon:

    - a situation never seen before (nothing to exploit yet)
    - an empty action set (nothing to do - escalate to a human)

Exploration can be backed by an `explorer` - any callable that takes a situation
and the available actions and returns one of them. Plug an LLM in there for a
warm cold-start (see oodaa/llm.py), or leave it None and explore at random. The
policy never imports a model itself; that seam stays outside the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

Explorer = Callable[[str, list], str]


@dataclass
class Decision:
    action: str
    reason: str
    kind: str = "act"  # act | done | escalate


class Policy:
    def __init__(self, epsilon: float = 0.1, explorer: Optional[Explorer] = None):
        self.epsilon = epsilon
        self.explorer = explorer

    def decide(self, obs, memory, rng) -> Decision:
        if obs.done:
            return Decision("", "objective met", kind="done")
        if not obs.actions:
            return Decision("", "no actions available", kind="escalate")

        explore = (not memory.seen(obs.situation, obs.actions)) or rng.random() < self.epsilon
        if explore:
            if self.explorer is not None:
                suggestion = self.explorer(obs.situation, obs.actions)
                if suggestion in obs.actions:
                    return Decision(suggestion, "explore (model suggestion)")
            return Decision(rng.choice(obs.actions), "explore (random)")

        action = memory.best(obs.situation, obs.actions)
        return Decision(action, f"exploit (value={memory.value(obs.situation, action):.2f})")
