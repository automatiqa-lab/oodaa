"""offline_grid - the proof, with no API key.

A 4x4 grid. The agent starts top-left, the goal is bottom-right. Every turn it
picks a direction; bumping a wall costs reward, moving toward the goal earns a
little, reaching it earns a lot. Nothing here is clever - the only learning is
the one-line Adjust update in oodaa/memory.py.

Run it and watch the steps-per-episode fall. Episode 1 wanders. By the end the
loop walks something close to the shortest path (6 steps on a 4x4). That drop is
the whole thesis: OODA would keep wandering quickly; OODAA gets shorter.

    python examples/offline_grid.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oodaa import Loop, Memory, Observation, Outcome, Policy  # noqa: E402


class GridWorld:
    """A deterministic grid task. situation is the agent's "x,y" coordinate."""

    SIZE = 4
    MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

    def __init__(self):
        self.x, self.y = 0, 0
        self.goal = (self.SIZE - 1, self.SIZE - 1)

    def observe(self) -> Observation:
        done = (self.x, self.y) == self.goal
        return Observation(situation=f"{self.x},{self.y}", actions=list(self.MOVES), done=done)

    def act(self, action: str) -> Outcome:
        dx, dy = self.MOVES[action]
        nx, ny = self.x + dx, self.y + dy
        if not (0 <= nx < self.SIZE and 0 <= ny < self.SIZE):
            return Outcome(reward=-1.0, success=False, note="wall")
        self.x, self.y = nx, ny
        if (self.x, self.y) == self.goal:
            return Outcome(reward=10.0, success=True, note="goal")
        dist = abs(self.goal[0] - self.x) + abs(self.goal[1] - self.y)
        # small step cost that grows with distance: nudges toward the goal
        return Outcome(reward=-0.1 - 0.05 * dist, success=True, note=f"dist={dist}")


def main() -> None:
    memory = Memory(lr=0.5)
    policy = Policy(epsilon=0.5)
    loop = Loop(policy=policy, memory=memory, max_steps=100, seed=7)

    episodes = 60
    lengths = []
    print("episode  steps  (lower is smarter)")
    for e in range(1, episodes + 1):
        # decay exploration: explore early, exploit what we learned later
        policy.epsilon = max(0.02, 0.5 * (0.92 ** e))
        result = loop.run(GridWorld())
        lengths.append(result.episode.length)
        if e <= 3 or e % 10 == 0:
            print(f"{e:>7}  {result.episode.length:>5}")

    first = sum(lengths[:5]) / 5
    last = sum(lengths[-5:]) / 5
    print(f"\nfirst 5 episodes averaged {first:.1f} steps")
    print(f"last 5 episodes averaged  {last:.1f} steps")
    print(f"shortest possible path is {2 * (GridWorld.SIZE - 1)} steps")
    print("\nthe loop got shorter because Adjust kept score. that is the second A.")


if __name__ == "__main__":
    main()
