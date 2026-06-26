"""The loop's control flow: it terminates, it gates on the objective, it escalates."""

from __future__ import annotations

from oodaa import Loop, Observation, Outcome, Policy


class AlreadyDone:
    def observe(self):
        return Observation(situation="start", actions=["noop"], done=True)

    def act(self, action):
        raise AssertionError("act must not be called once the objective is met")


class NoActions:
    def observe(self):
        return Observation(situation="stuck", actions=[], done=False)

    def act(self, action):
        raise AssertionError("act must not be called when there are no actions")


class NeverDone:
    """Always offers one action, never reports done - forces the step ceiling."""

    def observe(self):
        return Observation(situation="x", actions=["go"], done=False)

    def act(self, action):
        return Outcome(reward=0.0, success=True)


def test_done_objective_solves_without_acting():
    result = Loop().run(AlreadyDone())
    assert result.status == "solved"


def test_empty_action_set_escalates():
    result = Loop().run(NoActions())
    assert result.status == "escalated"


def test_step_ceiling_exhausts():
    result = Loop(max_steps=5).run(NeverDone())
    assert result.status == "exhausted"
    assert result.episode.length == 5


def test_phase_hook_sees_all_five_phases():
    seen = []
    loop = Loop(policy=Policy(epsilon=0.0), max_steps=3,
                on_phase=lambda phase, detail: seen.append(phase))
    loop.run(NeverDone())
    assert {"observe", "orient", "decide", "act", "adjust"} <= set(seen)
