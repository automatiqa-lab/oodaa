"""A tiny tool registry for the Act phase.

The loop calls Task.act(action). One common way to build a task is to back each
action by a function and let a registry route to it - that keeps the loop blind
to what the actions actually do (call an API, write a file, move a robot). The
ops example uses this; the grid example does not need it.
"""

from __future__ import annotations

from typing import Callable

from .state import Outcome


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable[..., Outcome]] = {}

    def register(self, name: str, fn: Callable[..., Outcome]) -> Callable[..., Outcome]:
        self._tools[name] = fn
        return fn

    def tool(self, name: str):
        """Decorator form: @registry.tool("reroute")."""

        def deco(fn: Callable[..., Outcome]) -> Callable[..., Outcome]:
            return self.register(name, fn)

        return deco

    def run(self, name: str, **kwargs) -> Outcome:
        fn = self._tools.get(name)
        if fn is None:
            return Outcome(reward=-1.0, success=False, note=f"unknown tool: {name}")
        return fn(**kwargs)

    @property
    def names(self) -> list:
        return list(self._tools)
