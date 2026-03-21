from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, status

from tripletex_agent.attachments import materialize_attachments
from tripletex_agent.config import get_settings
from tripletex_agent.executor import PlanExecutionError, PlanExecutor
from tripletex_agent.models import ExecutionPlan, ExecutionReport, SolveRequest, SolveResponse
from tripletex_agent.planner import PlanningError, build_planner
from tripletex_agent.tripletex_client import TripletexClient

logger = logging.getLogger("tripletex_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="AI-NM26 Tripletex Agent", version="0.1.0")


@app.on_event("startup")
def log_startup_metadata() -> None:
    settings = get_settings()
    logger.info(
        "Application startup: revision=%s planner_mode=%s beta_endpoints=%s timeout_seconds=%s",
        settings.app_revision,
        settings.planner_mode,
        settings.allow_beta_endpoints,
        settings.tripletex_timeout_seconds,
    )


@app.get("/")
def root() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "revision": settings.app_revision}


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "revision": settings.app_revision}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "revision": settings.app_revision}


@app.post("/solve", response_model=SolveResponse)
def solve(
    payload: SolveRequest,
    authorization: str | None = Header(default=None),
) -> SolveResponse:
    settings = get_settings()
    enforce_api_key(settings.endpoint_api_key, authorization)
    run_dir = create_run_dir(settings.runs_dir)
    run_id = run_dir.name
    logger.info(
        "Received solve request: run_id=%s revision=%s prompt=%r files=%d",
        run_id,
        settings.app_revision,
        payload.prompt[:400],
        len(payload.files),
    )
    attachments_dir = run_dir / "attachments"
    prepared_attachments = materialize_attachments(payload.files, attachments_dir)

    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "revision": settings.app_revision,
            "planner_mode": settings.planner_mode,
            "allow_beta_endpoints": settings.allow_beta_endpoints,
        },
    )
    write_json(run_dir / "request.json", payload.redacted_for_disk())
    write_json(
        run_dir / "attachments.json",
        [attachment.model_dump(mode="json") for attachment in prepared_attachments],
    )

    planner = build_planner(settings.planner_mode)

    try:
        plan = planner.build_plan(payload.prompt, prepared_attachments)
        logger.info(
            "Compiled plan: run_id=%s revision=%s goal=%r actions=%d",
            run_id,
            settings.app_revision,
            plan.goal,
            len(plan.actions),
        )
        write_json(run_dir / "plan.json", plan.model_dump(mode="json"))

        report = execute_plan(payload, plan, settings.tripletex_timeout_seconds)
        write_json(run_dir / "result.json", report.model_dump(mode="json"))
    except PlanningError as exc:
        persist_error(run_dir, exc, phase="planning")
        logger.warning(
            "Planning failed: run_id=%s revision=%s error=%s",
            run_id,
            settings.app_revision,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        persist_error(run_dir, exc, phase="unexpected")
        logger.exception(
            "Unhandled error while solving Tripletex task: run_id=%s revision=%s",
            run_id,
            settings.app_revision,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unhandled server error",
        ) from exc

    if report.error_message:
        persist_execution_report_error(run_dir, report)
        logger.warning(
            "Execution stopped early: run_id=%s revision=%s successful_actions=%d failed_action=%s error=%s",
            run_id,
            settings.app_revision,
            len(report.action_results),
            report.failed_action_id,
            report.error_message,
        )

    return SolveResponse(status="completed")


def execute_plan(payload: SolveRequest, plan: ExecutionPlan, timeout_seconds: float) -> ExecutionReport:
    with TripletexClient(
        base_url=payload.tripletex_credentials.base_url,
        session_token=payload.tripletex_credentials.session_token,
        timeout_seconds=timeout_seconds,
    ) as client:
        executor = PlanExecutor(client)
        return executor.execute(plan)


def enforce_api_key(expected_api_key: str | None, authorization_header: str | None) -> None:
    if not expected_api_key:
        return

    expected_header = f"Bearer {expected_api_key}"
    if authorization_header != expected_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


def create_run_dir(root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / f"{timestamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def persist_error(run_dir: Path, exc: Exception, *, phase: str) -> None:
    payload = {
        "phase": phase,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    write_json(run_dir / "error.json", payload)


def persist_execution_report_error(run_dir: Path, report: ExecutionReport) -> None:
    payload = {
        "phase": "execution",
        "error_type": report.error_type,
        "message": report.error_message,
        "failed_action_id": report.failed_action_id,
        "successful_actions": len(report.action_results),
    }
    write_json(run_dir / "error.json", payload)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
