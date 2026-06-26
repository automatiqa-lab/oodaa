<div align="center">

# oodaa

**A small, readable self-improving agent loop.**

Observe, Orient, Decide, Act - and then the second A, Adjust. The phase that makes an agent get smarter across runs instead of only faster within one.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

</div>

---

## Why this exists

Boyd's OODA loop was built for fighter pilots. Observe, Orient, Decide, Act, repeat - whoever cycles faster controls the engagement. It is the right model for a human who learns between cycles without thinking about it.

Agents do not. An agent that runs OODA makes faster decisions, not better ones - it repeats the same mistake at machine speed because nothing carries the lesson forward. The fix is one more phase. After acting, **Adjust**: score what happened and update how you decide next time. OODA gets faster. OODAA gets smarter. That difference is the whole repo.

The argument in long form: [Boyd's Loop Was Built for Pilots, Agentic AI Needs OODAA](https://alxsidr.io/boyds-loop-was-built-for-pilots-agentic-ai-needs-oodaa/). This is that essay made runnable - small enough to read in one sitting, real enough to build on.

## See it learn

No API key needed. A 4x4 grid, an agent that starts knowing nothing:

```bash
git clone https://github.com/automatiqa-lab/oodaa.git
cd oodaa
python examples/offline_grid.py
```

```
episode  steps  (lower is smarter)
      1     69
      2     31
      3     14
     30      6
     60      6

first 5 episodes averaged 27.2 steps
last 5 episodes averaged  6.0 steps
shortest possible path is 6 steps
```

Episode one wanders for 69 steps. By the end it walks the shortest path. The only thing that changed is what Adjust wrote to memory. Nothing in the loop is clever; the learning is a single line in [`oodaa/memory.py`](oodaa/memory.py).

## The five phases, mapped to the code

The loop is domain-neutral. It knows nothing about grids or shipments - a `Task` supplies the world, the loop supplies the cycle.

| Phase | What it does | Where it lives |
|-------|--------------|----------------|
| **Observe** | Ask the task what the world looks like now | `Task.observe()` -> [`state.py`](oodaa/state.py) `Observation` |
| **Orient** | Read what memory already knows about this situation | [`loop.py`](oodaa/loop.py) `_known`, [`memory.py`](oodaa/memory.py) |
| **Decide** | Pick an action, conditioned on memory | [`policy.py`](oodaa/policy.py) `Policy.decide` |
| **Act** | Run the action, get an outcome | `Task.act()`, optional [`executor.py`](oodaa/executor.py) |
| **Adjust** | Fold the outcome back into memory | [`memory.py`](oodaa/memory.py) `Memory.update` |

[`loop.py`](oodaa/loop.py) is the file to open first. Five blocks, five phases, one screen.

```python
from oodaa import Loop, Policy, Memory

loop = Loop(policy=Policy(epsilon=0.2), memory=Memory(lr=0.5))
result = loop.run(my_task)        # run it again and it decides better
```

## A supply-chain example

`oodaa` was built for the [automati.qa](https://www.automati.qa) lab, where the loops drive operational decisions. [`examples/ops_triage.py`](examples/ops_triage.py) runs the same machinery on a stream of incidents - a TMS shipment past its SLA, an ERP invoice blocked on a price variance, a WMS count gone negative. Each has a right first response the agent has to learn.

```bash
python examples/ops_triage.py
```

```
first 20 incidents: 50% right
last 20 incidents:  95% right
```

When the agent meets an incident type it has not seen, it can ask a language model what to try (the explorer in [`oodaa/llm.py`](oodaa/llm.py)). Every outcome feeds Adjust, so the next time that incident shows up, memory answers and the model is not needed. The example runs offline by default; wire any model in through `from_completion` and only the cold-start guess changes.

## Design choices

- **The core is the standard library.** No Redis, no database, no service. Clone and run.
- **The model is a seam, not a dependency.** The loop never imports a model. [`llm.py`](oodaa/llm.py) gives you an offline explorer and a `from_completion` seam that wraps any LLM in one line; either plugs into the policy, and a failed model call degrades to random exploration rather than taking the loop down.
- **Memory is plain and inspectable.** A running value per (situation, action), plus hypotheses you can attach. Print it and read what the loop believes. No black box.
- **Adjust is the point.** Everything else is scaffolding around the one phase that learns.

## Install

```bash
pip install -e .            # core, zero dependencies
pip install -e ".[dev]"     # add pytest
pytest
```

## Where it goes from here

This is the teaching-sized version. The production-grade version of the same idea - utility scoring, replay, a background self-review that proposes what to keep, gated self-modification - lives in [Synthax](https://github.com/alxsidr/synthax), where OODAA is the only loop the whole system runs. Start here to understand the shape; go there to see it carry real weight.

## License

MIT. See [LICENSE](LICENSE).
