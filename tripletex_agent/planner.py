from __future__ import annotations

import json

from pydantic import ValidationError

from tripletex_agent.gemini_planner import GeminiVertexPlanner
from tripletex_agent.models import ExecutionPlan, PreparedAttachment
from tripletex_agent.planning_base import PlanningError, TaskPlanner


class StubPlanner(TaskPlanner):
    def build_plan(self, prompt: str, attachments: list[PreparedAttachment]) -> ExecutionPlan:
        raise PlanningError(
            "No competition planner is configured yet. "
            "This scaffold separates prompt interpretation from execution so we can "
            "plug in an LLM-backed multilingual planner next."
        )


class JsonPromptPlanner(TaskPlanner):
    """
    Local development helper.

    If the prompt starts with PLAN_JSON:, the remainder must be a JSON object that
    matches the ExecutionPlan schema. This lets us test the executor against the
    real Tripletex proxy without pretending the multilingual planner is finished.
    """

    prefix = "PLAN_JSON:"

    def build_plan(self, prompt: str, attachments: list[PreparedAttachment]) -> ExecutionPlan:
        if not prompt.startswith(self.prefix):
            raise PlanningError("JsonPromptPlanner requires prompts that start with PLAN_JSON:")

        plan_payload = prompt[len(self.prefix) :].strip()
        try:
            raw = json.loads(plan_payload)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Invalid JSON plan: {exc}") from exc

        try:
            return ExecutionPlan.model_validate(raw)
        except ValidationError as exc:
            raise PlanningError(f"Plan does not match ExecutionPlan schema: {exc}") from exc


def build_planner(mode: str) -> TaskPlanner:
    if mode == "json_prompt":
        return JsonPromptPlanner()
    if mode == "gemini_vertex":
        from tripletex_agent.config import get_settings

        settings = get_settings()
        return GeminiVertexPlanner(
            api_key=settings.gemini_api_key,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            model=settings.gemini_model,
            max_attachment_bytes=settings.gemini_max_attachment_bytes,
            attachment_text_chars=settings.gemini_attachment_text_chars,
            allow_beta_endpoints=settings.allow_beta_endpoints,
        )
    return StubPlanner()
