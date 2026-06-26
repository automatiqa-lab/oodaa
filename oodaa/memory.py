"""Memory - the part that learns.

This is where Adjust writes and Decide reads. It is deliberately a plain,
inspectable structure: a running value estimate per (situation, action), plus a
list of hypotheses you can attach by hand or from a review step. Print it and
you can read exactly what the loop believes.

The value update is the smallest thing that counts as learning:

    value <- value + lr * (reward - value)

an incremental average that drifts toward whatever reward an action actually
earns in a given situation. Synthax does the heavyweight version of this
(utility scores, replay, a background self-review that proposes what to keep);
this is that idea with the machinery taken out so you can see the shape of it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ActionStats:
    tries: int = 0
    wins: int = 0
    value: float = 0.0

    @property
    def confidence(self) -> float:
        return self.wins / self.tries if self.tries else 0.0


@dataclass
class Memory:
    lr: float = 0.5  # learning rate: how fast new evidence overrides old belief
    stats: dict = field(default_factory=lambda: defaultdict(ActionStats))
    hypotheses: list = field(default_factory=list)

    def value(self, situation: str, action: str) -> float:
        return self.stats[(situation, action)].value

    def tried(self, situation: str, action: str) -> int:
        return self.stats[(situation, action)].tries

    def seen(self, situation: str, actions: list) -> bool:
        """True once any action has been tried in this situation."""
        return any(self.stats[(situation, a)].tries for a in actions)

    def best(self, situation: str, actions: list) -> str:
        """The action with the highest learned value. Ties resolve to the
        first in the given order, so pass actions in a stable order."""
        return max(actions, key=lambda a: self.stats[(situation, a)].value)

    def update(self, situation: str, action: str, outcome) -> ActionStats:
        """Adjust. Fold one outcome into the estimate for this (situation, action)."""
        s = self.stats[(situation, action)]
        s.tries += 1
        if outcome.success:
            s.wins += 1
        s.value += self.lr * (outcome.reward - s.value)
        return s

    def note(self, hypothesis: str) -> None:
        """Attach a falsifiable note about why something works. Free-form, kept
        unique. A heavier loop would generate and test these automatically."""
        if hypothesis and hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
