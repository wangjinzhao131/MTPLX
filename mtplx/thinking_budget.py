"""Exact request-level thinking budgets, including speculative prefixes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from .thinking_guard import ThinkingGuard, ThinkingGuardConfig


@dataclass(frozen=True)
class HardThinkingBudgetConfig(ThinkingGuardConfig):
    """Opt into hard target masks through the existing thinking_guard slot."""


class HardThinkingBudget(ThinkingGuard):
    def observe(self, tokens: Sequence[int]) -> str | None:
        if len(tokens) < self._state.count:
            self.__init__(self.config)
        transition = None
        # MTP commits several tokens at once; engage at the exact token boundary.
        for count in range(self._state.count + 1, len(tokens) + 1):
            marker = super().observe(tokens[:count])
            # Keep the cumulative budget active if the model reopens thinking.
            self.dormant = False
            if marker is not None and marker != "dormant":
                transition = marker
        return transition

    @property
    def steering_active(self) -> bool:
        # A speculative prefix may cross the budget before the next observe().
        return self.config.enabled

    def overlay_for(self, working: Sequence[int]) -> None:
        return None

    def forced_token_for(self, working: Sequence[int]) -> int | None:
        if not self.config.enabled:
            return None
        preview = self
        if len(working) != self._state.count:
            preview = deepcopy(self)
            preview.observe(working)
        if preview.engaged is not None and not preview.forced_done:
            index = len(working) - int(preview.engaged_at or 0)
            if 0 <= index < len(self.config.forced_close_ids):
                return int(self.config.forced_close_ids[index])
        return None
