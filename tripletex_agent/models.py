from __future__ import annotations

import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TripletexCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    session_token: str

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")


class InboundFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    content_base64: str
    mime_type: str

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("filename cannot be empty")
        return value

    def decoded_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.content_base64, validate=True)
        except binascii.Error as exc:
            raise ValueError("content_base64 is not valid base64") from exc


class SolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    files: list[InboundFile] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be empty")
        return value

    def redacted_for_disk(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["tripletex_credentials"]["session_token"] = "<redacted>"
        for file_entry in payload["files"]:
            file_entry["content_base64"] = "<redacted>"
        return payload


class SolveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"]


class PreparedAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    mime_type: str
    saved_path: str
    size_bytes: int
    extracted_text: str | None = None


class TaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str
    method: Literal["GET", "POST", "PUT", "DELETE", "SELECT"]
    path: str
    params: dict[str, Any] = Field(default_factory=dict)
    body: Any = None
    save_as: str | None = None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    language_hint: str | None = None
    actions: list[TaskAction] = Field(default_factory=list)
    verification_notes: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    method: str
    path: str
    status_code: int
    response: Any


class ExecutionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_calls: int
    action_results: list[ActionResult] = Field(default_factory=list)
    saved_context: dict[str, Any] = Field(default_factory=dict)
