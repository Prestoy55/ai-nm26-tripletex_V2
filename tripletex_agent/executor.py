from __future__ import annotations

import re
from typing import Any

from tripletex_agent.models import ActionResult, ExecutionPlan, ExecutionReport
from tripletex_agent.tripletex_client import TripletexApiError, TripletexClient

_TEMPLATE = re.compile(r"{{\s*([^{}]+?)\s*}}")


class PlanExecutionError(RuntimeError):
    """Raised when a plan cannot be resolved or executed."""


class PlanExecutor:
    def __init__(self, client: TripletexClient) -> None:
        self.client = client

    def execute(self, plan: ExecutionPlan) -> ExecutionReport:
        context: dict[str, Any] = {}
        results: list[ActionResult] = []

        for action in plan.actions:
            try:
                resolved_path = resolve_templates(action.path, context)
                resolved_params = compact_payload(resolve_templates(action.params, context))
                resolved_body = compact_payload(resolve_templates(action.body, context))

                if action.method == "SELECT":
                    response_payload = run_select_action(action.id, resolved_body)
                    context[action.id] = response_payload
                    if action.save_as:
                        context[action.save_as] = response_payload
                    results.append(
                        ActionResult(
                            action_id=action.id,
                            method=action.method,
                            path="(local select)",
                            status_code=0,
                            response=response_payload,
                        )
                    )
                    continue

                if action.method == "ENSURE_ACCOUNT":
                    status_code, response_payload = run_ensure_account_action(
                        self.client,
                        action.id,
                        resolved_path,
                        resolved_body,
                    )
                    context[action.id] = response_payload
                    if action.save_as:
                        context[action.save_as] = response_payload
                    results.append(
                        ActionResult(
                            action_id=action.id,
                            method=action.method,
                            path=resolved_path,
                            status_code=status_code,
                            response=response_payload,
                        )
                    )
                    continue

                if not isinstance(resolved_path, str):
                    raise PlanExecutionError(f"Resolved path for action {action.id} must be a string")

                status_code, response_payload = self.client.request(
                    action.method,
                    resolved_path,
                    params=resolved_params,
                    json_body=resolved_body,
                )

                context[action.id] = response_payload
                if action.save_as:
                    context[action.save_as] = response_payload

                results.append(
                    ActionResult(
                        action_id=action.id,
                        method=action.method,
                        path=resolved_path,
                        status_code=status_code,
                        response=response_payload,
                    )
                )
            except (PlanExecutionError, TripletexApiError) as exc:
                return ExecutionReport(
                    api_calls=self.client.calls_made,
                    action_results=results,
                    saved_context=context,
                    failed_action_id=action.id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        return ExecutionReport(
            api_calls=self.client.calls_made,
            action_results=results,
            saved_context=context,
        )


def resolve_templates(value: Any, context: dict[str, Any]) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        match = _TEMPLATE.fullmatch(value)
        if match:
            return lookup_reference(match.group(1), context)

        return _TEMPLATE.sub(lambda matched: str(lookup_reference(matched.group(1), context)), value)

    if isinstance(value, list):
        return [resolve_templates(item, context) for item in value]

    if isinstance(value, dict):
        return {key: resolve_templates(item, context) for key, item in value.items()}

    return value


def lookup_reference(reference: str, context: dict[str, Any]) -> Any:
    current: Any = context
    for part in reference.split("."):
        if isinstance(current, dict):
            if part not in current:
                raise PlanExecutionError(f"Unknown template reference: {reference}")
            current = current[part]
            continue

        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise PlanExecutionError(f"Expected numeric list index in reference: {reference}") from exc

            try:
                current = current[index]
            except IndexError as exc:
                raise PlanExecutionError(f"List index out of range in reference: {reference}") from exc
            continue

        raise PlanExecutionError(f"Cannot traverse reference {reference} through {type(current).__name__}")

    return current


def compact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            resolved = compact_payload(item)
            if resolved is None:
                continue
            compacted[key] = resolved
        return compacted or None

    if isinstance(value, list):
        compacted_list = [resolved for item in value if (resolved := compact_payload(item)) is not None]
        return compacted_list

    return value


def run_select_action(action_id: str, payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise PlanExecutionError(f"SELECT action {action_id} requires an object body")

    source = payload.get("source")
    criteria = payload.get("criteria", {})

    if not isinstance(source, list):
        raise PlanExecutionError(f"SELECT action {action_id} requires body.source to be a list")
    if not isinstance(criteria, dict):
        raise PlanExecutionError(f"SELECT action {action_id} requires body.criteria to be an object")

    matches: list[Any] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        if all(values_match(select_value(item, key), expected) for key, expected in criteria.items()):
            matches.append(item)

    if not matches:
        raise PlanExecutionError(f"SELECT action {action_id} found no matching items")
    if len(matches) > 1:
        raise PlanExecutionError(f"SELECT action {action_id} found multiple matching items")
    return matches[0]


def run_ensure_account_action(
    client: TripletexClient,
    action_id: str,
    path: Any,
    payload: Any,
) -> tuple[int, Any]:
    if not isinstance(path, str):
        raise PlanExecutionError(f"ENSURE_ACCOUNT action {action_id} requires a string path")
    if not isinstance(payload, dict):
        raise PlanExecutionError(f"ENSURE_ACCOUNT action {action_id} requires an object body")

    source = payload.get("source")
    account_number = payload.get("number")
    create_body = payload.get("create_body")

    if not isinstance(source, list):
        raise PlanExecutionError(f"ENSURE_ACCOUNT action {action_id} requires body.source to be a list")
    if not isinstance(account_number, int):
        raise PlanExecutionError(f"ENSURE_ACCOUNT action {action_id} requires body.number to be an integer")
    if not isinstance(create_body, dict):
        raise PlanExecutionError(f"ENSURE_ACCOUNT action {action_id} requires body.create_body to be an object")

    matches = [
        item
        for item in source
        if isinstance(item, dict) and values_match(select_value(item, "number"), account_number)
    ]
    if len(matches) == 1:
        return 0, matches[0]
    if len(matches) > 1:
        raise PlanExecutionError(
            f"ENSURE_ACCOUNT action {action_id} found multiple matching accounts for {account_number}"
        )

    status_code, response_payload = client.request("POST", path, json_body=create_body)
    if isinstance(response_payload, dict) and isinstance(response_payload.get("value"), dict):
        return status_code, response_payload["value"]
    return status_code, response_payload


def select_value(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, (int, float, str)) and isinstance(expected, (int, float, str)):
        return str(actual).strip() == str(expected).strip()
    return False
