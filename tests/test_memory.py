"""Adjust actually adjusts: memory changes, and the loop gets measurably better."""

from __future__ import annotations

import sys
from pathlib import Path

from oodaa import Loop, Memory, Outcome, Policy

# Reuse the grid task from the example as a learning fixture.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from offline_grid import GridWorld  # noqa: E402


def test_update_moves_value_and_counts():
    m = Memory(lr=0.5)
    m.update("s", "a", Outcome(reward=10.0, success=True))
    stats = m.stats[("s", "a")]
    assert stats.tries == 1
    assert stats.wins == 1
    assert stats.value == 5.0  # 0 + 0.5 * (10 - 0)


def test_best_picks_highest_value():
    m = Memory(lr=1.0)
    m.update("s", "a", Outcome(reward=1.0, success=True))
    m.update("s", "b", Outcome(reward=9.0, success=True))
    assert m.best("s", ["a", "b"]) == "b"


def test_unseen_situation_is_flagged_for_exploration():
    m = Memory()
    assert not m.seen("fresh", ["a", "b"])
    m.update("fresh", "a", Outcome(reward=0.0, success=True))
    assert m.seen("fresh", ["a", "b"])


def test_loop_gets_shorter_over_episodes():
    """The point of the whole repo: steps-to-solve fall as Adjust accumulates."""
    policy = Policy(epsilon=0.5)
    loop = Loop(policy=policy, memory=Memory(lr=0.5), max_steps=100, seed=7)
    lengths = []
    for e in range(1, 61):
        policy.epsilon = max(0.02, 0.5 * (0.92 ** e))
        lengths.append(loop.run(GridWorld()).episode.length)

    first = sum(lengths[:5]) / 5
    last = sum(lengths[-5:]) / 5
    assert last < first  # it learned
    assert last <= 10  # and converged near the 6-step optimum
