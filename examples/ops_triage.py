"""ops_triage - the same loop, on a supply-chain decision, with a model in Decide.

A stream of operational incidents arrives - a TMS shipment past its SLA, an ERP
invoice blocked on a price variance, a WMS count that went negative. Each one has
a right first response. The agent does not know the mapping up front. When it
meets an incident type it has not seen, it asks a language model what to try (the
explorer in oodaa/llm.py). Every outcome feeds Adjust, so the second time that
incident type shows up, memory answers and the model is not needed.

This runs offline out of the box - the explorer picks at random until memory
takes over, and the loop still learns. To use a real model, wire one into
`make_explorer` below through `from_completion`; the loop is identical, only the
cold-start guess gets sharper. Either way the back half beats the front half.

    python examples/ops_triage.py
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oodaa import Loop, Memory, Observation, Outcome, Policy  # noqa: E402
from oodaa.llm import from_completion, offline  # noqa: E402, F401  (from_completion shown in make_explorer)

# Generic systems only - TMS, ERP, WMS. No vendor names.
INCIDENTS = {
    "TMS: shipment ETA slipped past the SLA window": "expedite",
    "ERP: invoice blocked on a price variance": "escalate_to_buyer",
    "WMS: cycle count returned negative stock": "hold_and_recount",
    "TMS: carrier tendered but no pickup scan in 24h": "reroute",
    "ERP: PO quantity does not match the goods receipt": "reconcile",
}
ACTIONS = ["expedite", "escalate_to_buyer", "hold_and_recount", "reroute", "reconcile"]


class OpsTriage:
    """One incident, one decision. success = the right first response was chosen."""

    def __init__(self, incident: str, correct: str):
        self.incident = incident
        self.correct = correct
        self._answered = False

    def observe(self) -> Observation:
        return Observation(situation=self.incident, actions=ACTIONS, done=self._answered)

    def act(self, action: str) -> Outcome:
        self._answered = True
        if action == self.correct:
            return Outcome(reward=1.0, success=True, note="correct")
        return Outcome(reward=-1.0, success=False, note=f"wrong (correct: {self.correct})")


def make_explorer():
    """Return the explorer Decide uses for unfamiliar situations.

    Offline by default so the example runs with no setup. To use a hosted model,
    fill in `complete` with a call to whatever LLM you run and return
    from_completion(complete) - nothing else changes.
    """
    # def complete(prompt: str) -> str:
    #     return my_model.generate(prompt)        # any provider, your call
    # return from_completion(complete)
    print("explorer: offline (wire a model into make_explorer to warm the cold-start)\n")
    return offline(seed=1)


def main() -> None:
    policy = Policy(epsilon=0.1, explorer=make_explorer())
    memory = Memory(lr=0.5)
    loop = Loop(policy=policy, memory=memory, max_steps=2, seed=3)

    items = list(INCIDENTS.items())
    stream = random.Random(0)
    correct = []
    for _ in range(40):
        incident, answer = stream.choice(items)
        result = loop.run(OpsTriage(incident, answer))
        correct.append(result.episode.steps[0].outcome.success)

    half = len(correct) // 2
    first = sum(correct[:half]) / half
    last = sum(correct[half:]) / (len(correct) - half)
    print(f"first {half} incidents: {first:.0%} right")
    print(f"last {len(correct) - half} incidents:  {last:.0%} right")
    print("\nmemory carried what worked. Decide stopped guessing. that is Adjust at work.")


if __name__ == "__main__":
    main()
