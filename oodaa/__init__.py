"""oodaa - a small, readable self-improving agent loop.

Observe, Orient, Decide, Act, Adjust. OODA stops at Act; the second A is what
makes the loop get smarter across runs instead of only faster within one.

Public surface:
    from oodaa import Loop, Policy, Memory, Observation, Outcome, ToolRegistry
"""

from .state import Episode, Observation, Outcome, Step
from .memory import Memory, ActionStats
from .policy import Policy, Decision
from .loop import Loop, RunResult, Task
from .executor import ToolRegistry

__all__ = [
    "Loop",
    "RunResult",
    "Task",
    "Policy",
    "Decision",
    "Memory",
    "ActionStats",
    "Episode",
    "Observation",
    "Outcome",
    "Step",
    "ToolRegistry",
]

__version__ = "0.1.0"
