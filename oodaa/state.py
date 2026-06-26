"""The data the loop moves around: what it sees, what it gets back, what it keeps.

Three small records and one container:
    Observation  what Observe hands to Decide - the situation and the legal moves
    Outcome      what Act returns and Adjust learns from - a reward and a verdict
    Step         one (situation, action, outcome) triple
    Episode      the working memory for a single run
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Observation:
    """A snapshot of the world for one turn.

    situation is a compact, hashable description - the loop uses it as a memory
    key, so keep it small and stable (a coordinate, an incident type, a state
    signature). actions is the set Decide is allowed to choose from right now.
    """

    situation: str
    actions: list[str]
    done: bool = False
    detail: dict = field(default_factory=dict)


@dataclass
class Outcome:
    """The result of one action.

    reward is the only signal the learner needs - higher is better. success is a
    human-readable verdict used for reporting and confidence, not for the value
    update. note is free text for tracing.
    """

    reward: float
    success: bool
    note: str = ""


@dataclass
class Step:
    situation: str
    action: str
    outcome: Outcome


@dataclass
class Episode:
    """Everything that happened in one run of the loop."""

    steps: list[Step] = field(default_factory=list)
    status: str = "running"  # running | solved | exhausted | escalated

    def record(self, situation: str, action: str, outcome: Outcome) -> None:
        self.steps.append(Step(situation, action, outcome))

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def reward(self) -> float:
        return sum(s.outcome.reward for s in self.steps)
