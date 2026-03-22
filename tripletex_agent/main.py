from __future__ import annotations

import json
import logging
import re
import unicodedata
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
    prompt_family = classify_prompt_family(payload.prompt, len(payload.files))
    logger.info(
        "Received solve request: run_id=%s revision=%s family=%s prompt=%r files=%d",
        run_id,
        settings.app_revision,
        prompt_family,
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
            "Compiled plan: run_id=%s revision=%s family=%s goal=%r actions=%d",
            run_id,
            settings.app_revision,
            prompt_family,
            plan.goal,
            len(plan.actions),
        )
        write_json(run_dir / "plan.json", plan.model_dump(mode="json"))

        report = execute_plan(payload, plan, settings.tripletex_timeout_seconds)
        write_json(run_dir / "result.json", report.model_dump(mode="json"))
    except PlanningError as exc:
        persist_error(run_dir, exc, phase="planning")
        logger.warning(
            "Planning failed: run_id=%s revision=%s family=%s error=%s",
            run_id,
            settings.app_revision,
            prompt_family,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        persist_error(run_dir, exc, phase="unexpected")
        logger.exception(
            "Unhandled error while solving Tripletex task: run_id=%s revision=%s family=%s",
            run_id,
            settings.app_revision,
            prompt_family,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unhandled server error",
        ) from exc

    if report.error_message:
        persist_execution_report_error(run_dir, report)
        logger.warning(
            "Execution stopped early: run_id=%s revision=%s family=%s successful_actions=%d failed_action=%s error=%s",
            run_id,
            settings.app_revision,
            prompt_family,
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


def classify_prompt_family(prompt: str, file_count: int) -> str:
    normalized = normalize_text(prompt)

    if any(
        token in normalized
        for token in (
            "fri rekneskapsdimensjon",
            "free accounting dimension",
            "custom accounting dimension",
            "dimension libre",
            "dimension contable",
        )
    ):
        return "custom_accounting_dimension"
    if file_count and any(
        token in normalized
        for token in (
            "bankutskrifta",
            "bank statement",
            "bank statement",
            "bankutskrift",
            "extrato bancario",
            "releve bancaire",
        )
    ):
        return "bank_reconciliation"
    if file_count and any(
        token in normalized
        for token in (
            "contract",
            "contrato de trabajo",
            "arbeidskontrakt",
            "contrat de travail",
            "carta de oferta",
            "arbeidstilbud",
        )
    ):
        return "employee_from_document"
    if any(
        token in normalized
        for token in (
            "new employee",
            "ny ansatt",
            "neuen mitarbeiter",
            "nuevo empleado",
            "nouvel employe",
            "creer en tant qu'employe",
            "veuillez le creer en tant qu'employe",
            "legg ham som medarbeider",
        )
    ) or re.search(r"(novo funcion.rio|crie-o como funcion.rio)", normalized):
        return "employee_create"
    if any(
        token in normalized
        for token in (
            "supplier invoice",
            "lieferantenrechnung",
            "leverandorfaktura",
            "facture fournisseur",
            "factura del proveedor",
            "fatura do fornecedor",
            "vom lieferanten",
            "from the supplier",
        )
    ):
        return "supplier_invoice"
    if any(
        token in normalized
        for token in (
            "register supplier",
            "registrer leverandor",
            "registrer leverandor",
            "registrieren sie den lieferanten",
            "registrer leverandoren",
            "registre el proveedor",
            "registre o fornecedor",
            "registrer leverandøren",
        )
    ):
        return "customer_create"
    if any(token in normalized for token in ("create product", "opprett produkt", "creez le produit", "crie o produto", "producto")):
        return "product_create"
    if any(token in normalized for token in ("create three departments", "opprett tre avdel", "cree trois depart", "departments in tripletex", "avdelingar", "departamentos")):
        return "department_batch_create"
    if any(
        token in normalized
        for token in (
            "project manager",
            "prosjektleder",
            "projektleiter",
            "director del proyecto",
            "gerente de projeto",
            "create project",
            "opprett prosjekt",
            "erstellen sie das projekt",
            "cycle de vie complet du projet",
            "projet '",
            'projet "',
        )
    ):
        return "project_create"
    if any(
        token in normalized
        for token in (
            "payroll",
            "salary",
            "bonus",
            "gehaltsabrechnung",
            "grundgehalt",
            "gehalt",
            "lonn",
            "salario",
        )
    ):
        return "payroll_voucher"
    if re.search(
        r"(credit note|credit memo|kreditnota|kreditnote|nota de cr.?dito|note de cr.?dit)",
        normalized,
    ):
        return "credit_note"
    if any(token in normalized for token in ("register full payment", "registrer full betaling", "zahlung", "pagamento", "payment was returned", "reverser betalingen")):
        return "invoice_payment_flow"
    if any(token in normalized for token in ("exchange rate", "wechselkurs", "disagio", "agio")):
        return "foreign_currency_payment"
    if any(token in normalized for token in ("find de 4 errors", "finn de 4 feilene", "revise todos os vouchers", "hauptbuch", "livro razao")):
        return "ledger_correction"
    if any(
        token in normalized
        for token in (
            "travel expense",
            "reisekost",
            "reiseregning",
            "despesa de viagem",
            "frais de deplacement",
        )
    ):
        return "travel_expense"
    if file_count and any(token in normalized for token in ("receipt", "recibo", "kvittering")):
        return "receipt_voucher"
    if any(token in normalized for token in ("invoice", "faktura", "facture", "factura", "fatura")):
        return "invoice_create"
    if any(token in normalized for token in ("customer", "kunde", "cliente", "client")):
        return "customer_create"
    return "unknown"


def normalize_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_value).strip().lower()
