from __future__ import annotations

from typing import Any

import httpx


class TripletexApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, response_body: Any):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class TripletexClient:
    def __init__(self, *, base_url: str, session_token: str, timeout_seconds: float) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            auth=("0", session_token),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
        )
        self.calls_made = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TripletexClient":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> tuple[int, Any]:
        normalized_path = path if path.startswith("/") else f"/{path}"
        response = self._client.request(method, normalized_path, params=params, json=json_body)
        self.calls_made += 1

        payload = parse_response_body(response)
        if response.is_error:
            detail = summarize_tripletex_error(payload)
            message = f"Tripletex returned HTTP {response.status_code} for {method} {normalized_path}"
            if detail:
                message = f"{message}: {detail}"
            raise TripletexApiError(
                message,
                status_code=response.status_code,
                response_body=payload,
            )

        return response.status_code, payload


def parse_response_body(response: httpx.Response) -> Any:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.text


def summarize_tripletex_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    messages: list[str] = []
    top_message = payload.get("message")
    if isinstance(top_message, str) and top_message.strip():
        messages.append(top_message.strip())

    validation_messages = payload.get("validationMessages")
    if isinstance(validation_messages, list):
        for item in validation_messages[:3]:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            message = item.get("message")
            if isinstance(field, str) and field.strip() and isinstance(message, str) and message.strip():
                messages.append(f"{field}: {message.strip()}")
            elif isinstance(message, str) and message.strip():
                messages.append(message.strip())

    if not messages:
        return None

    deduped: list[str] = []
    for item in messages:
        if item not in deduped:
            deduped.append(item)
    return " | ".join(deduped)
