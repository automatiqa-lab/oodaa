"""The OODAA loop.

One run = one Episode. Each turn walks the five phases:

    Observe   ask the task what the world looks like now
    Orient    read what memory already knows about this situation
    Decide    the policy picks an action, conditioned on that memory
    Act        the task executes the action and returns an Outcome
    Adjust     fold the Outcome back into memory so the next run decides better

OODA - Boyd's loop - stops at Act. It makes a pilot faster. The second A,
Adjust, is what makes an agent smarter: without it the loop repeats its mistakes
at speed. That phase is one line here (`memory.update`), and it is the whole
reason the repo exists. Run the same task many times and watch the behaviour
change - that is the demo in examples/offline_grid.py.

This module is domain-neutral. It knows nothing about grids, incidents, or tools.
A `Task` supplies the world; the loop supplies the cycle around it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .memory import Memory
from .policy import Policy
from .state import Episode


class Task(Protocol):
    """The world the loop drives. Implement these two methods and the loop runs."""

    def observe(self): ...  # -> Observation

    def act(self, action: str): ...  # -> Outcome


@dataclass
class RunResult:
    status: str  # solved | exhausted | escalated
    episode: Episode
    reason: str = ""


class Loop:
    def __init__(
        self,
        policy: Optional[Policy] = None,
        memory: Optional[Memory] = None,
        max_steps: int = 50,
        seed: int = 0,
        on_phase: Optional[Callable[[str, dict], None]] = None,
    ):
        self.policy = policy or Policy()
        self.memory = memory or Memory()
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.on_phase = on_phase  # optional trace hook: (phase_name, detail) -> None

    def run(self, task: Task) -> RunResult:
        episode = Episode()
        for step in range(1, self.max_steps + 1):
            # OBSERVE
            obs = task.observe()
            self._emit("observe", {"step": step, "situation": obs.situation})

            # ORIENT - what does memory know about the moves available here
            self._emit("orient", {"known": self._known(obs)})

            # DECIDE
            decision = self.policy.decide(obs, self.memory, self.rng)
            self._emit("decide", {"action": decision.action, "reason": decision.reason})
            if decision.kind == "done":
                episode.status = "solved"
                return RunResult("solved", episode, decision.reason)
            if decision.kind == "escalate":
                episode.status = "escalated"
                return RunResult("escalated", episode, decision.reason)

            # ACT
            outcome = task.act(decision.action)
            self._emit("act", {"action": decision.action, "reward": outcome.reward,
                               "success": outcome.success})

            # ADJUST - the second A. Update memory, then record the step.
            self.memory.update(obs.situation, decision.action, outcome)
            episode.record(obs.situation, decision.action, outcome)
            self._emit("adjust", {"value": self.memory.value(obs.situation, decision.action)})

        episode.status = "exhausted"
        return RunResult("exhausted", episode, "hit the step ceiling")

    def _known(self, obs) -> dict:
        return {a: round(self.memory.value(obs.situation, a), 2) for a in obs.actions}

    def _emit(self, phase: str, detail: dict) -> None:
        if self.on_phase is not None:
            self.on_phase(phase, detail)
