from __future__ import annotations

from abc import ABC, abstractmethod

from tripletex_agent.models import ExecutionPlan, PreparedAttachment


class PlanningError(RuntimeError):
    """Raised when the prompt cannot be transformed into an execution plan."""


class TaskPlanner(ABC):
    @abstractmethod
    def build_plan(self, prompt: str, attachments: list[PreparedAttachment]) -> ExecutionPlan:
        raise NotImplementedError
