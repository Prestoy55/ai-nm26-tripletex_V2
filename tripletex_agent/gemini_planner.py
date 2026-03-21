from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

from tripletex_agent.models import ExecutionPlan, PreparedAttachment
from tripletex_agent.planning_base import PlanningError, TaskPlanner
from tripletex_agent.task_compiler import compile_task_intent
from tripletex_agent.task_intents import SUPPORTED_TASK_INTENT_ADAPTER

SYSTEM_INSTRUCTION = """
You are the planning layer for a Tripletex accounting agent in the Norwegian AI Championship 2026.

Your job is to transform a multilingual accounting prompt and any attachments into a compact supported task
intent JSON object that matches the provided schema exactly.

Operational rules:
- The downstream compiler is deterministic and will transform your supported task intent into canonical API calls.
- The Tripletex account starts empty for each competition submission.
- Return only JSON matching the supported task intent schema.
- Never include markdown, commentary, or code fences.
- Only return task_type=unsupported as a last resort.
- If you return task_type=unsupported, you must include a short non-empty reason string.

Task guidance:
- Prompts may arrive in Norwegian Bokmal, English, Spanish, Portuguese, Nynorsk, German, or French.
- Extract exact names, dates, amounts, email addresses, phone numbers, customer names, and entity relations.
- If attachments contain relevant task data, incorporate it into the intent.
- The account starts empty, so create tasks should assume resources do not already exist.

Currently supported task families:
- create_employee
- create_customer
- update_customer
- create_product
- update_employee
- create_department
- delete_department
- create_travel_expense
- delete_travel_expense
- create_project
- create_invoice

Important mapping rules:
- If the user asks for an account administrator, administrator, or full rights, set entitlement_template=ALL_PRIVILEGES and user_type=EXTENDED.
- If the user asks for invoicing manager rights, use entitlement_template=INVOICING_MANAGER and user_type=EXTENDED.
- If the user asks for HR/personnel manager rights, use entitlement_template=PERSONELL_MANAGER and user_type=EXTENDED.
- If the user asks to update a customer, use update_customer and match by the most specific identifier available: organization number, email, then name.
- If the user asks to update an employee, use update_employee and match by employee number or email when available, otherwise by first and last name.
- If the user asks to delete a department, use delete_department and set match_name to the department name.
- If the user asks to create a travel expense, use create_travel_expense. Extract the title plus employee first and last name. Include details when the prompt provides trip dates, origin, destination, purpose, or whether it is a day trip or foreign travel.
- If the user asks to delete a travel expense, use delete_travel_expense and extract the travel expense title plus the employee first and last name.
- If the prompt includes an employee birth date for an update, put it in new_date_of_birth using ISO format.
- If the user asks to create a project for a customer in an otherwise empty account, include the customer subobject so the compiler can create the customer first.
- For invoices, extract customer details and invoice lines. The compiler will create customer, order, and invoice in that order.
- For invoices, default send_to_customer=false unless the prompt explicitly says to send, email, dispatch, or deliver the invoice to the customer.
- If the user asks to register payment for an invoice, still use create_invoice and set register_full_payment=true.
- For customer creation, flat address fields like address, postal_code, and city should be mapped into postal_address.
- Invoice lines must include description and quantity. Include either unit_price_excluding_vat_currency or unit_price_including_vat_currency when the prompt provides price information.
- Use ISO dates in YYYY-MM-DD format.
""".strip()


class GeminiVertexPlanner(TaskPlanner):
    def __init__(
        self,
        *,
        api_key: str | None,
        project: str | None,
        location: str,
        model: str,
        max_attachment_bytes: int,
        attachment_text_chars: int,
        allow_beta_endpoints: bool,
    ) -> None:
        self.api_key = api_key
        self.project = project
        self.location = location
        self.model = model
        self.max_attachment_bytes = max_attachment_bytes
        self.attachment_text_chars = attachment_text_chars
        self.allow_beta_endpoints = allow_beta_endpoints

    def build_plan(self, prompt: str, attachments: list[PreparedAttachment]) -> ExecutionPlan:
        if self.api_key:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=60_000),
            )
        else:
            client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options=types.HttpOptions(api_version="v1", timeout=60_000),
            )
        contents = self._build_contents(prompt, attachments)
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                systemInstruction=SYSTEM_INSTRUCTION,
                temperature=0,
                responseMimeType="application/json",
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise PlanningError("Gemini planner returned an empty response")

        try:
            raw_intent = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Gemini planner returned invalid JSON: {exc}") from exc

        normalized_intent = normalize_intent_payload(raw_intent)

        try:
            intent = SUPPORTED_TASK_INTENT_ADAPTER.validate_python(normalized_intent)
        except Exception as exc:
            raise PlanningError(
                "Gemini planner returned invalid task intent JSON: "
                f"{exc}. Raw JSON: {truncate_text(text, limit=1200)}"
            ) from exc

        return compile_task_intent(intent, allow_beta_endpoints=self.allow_beta_endpoints)

    def _build_contents(self, prompt: str, attachments: list[PreparedAttachment]) -> list[object]:
        payload = {
            "today": date.today().isoformat(),
            "task_prompt": prompt,
            "attachments": [
                {
                    "filename": attachment.filename,
                    "mime_type": attachment.mime_type,
                    "saved_path": attachment.saved_path,
                    "size_bytes": attachment.size_bytes,
                    "extracted_text": truncate_text(
                        attachment.extracted_text,
                        limit=self.attachment_text_chars,
                    ),
                }
                for attachment in attachments
            ],
        }

        contents: list[object] = [
            "Build an ExecutionPlan JSON object for the attached Tripletex task.",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]

        for attachment in attachments:
            if not should_inline_attachment(
                attachment.mime_type,
                attachment.size_bytes,
                max_attachment_bytes=self.max_attachment_bytes,
            ):
                continue

            contents.append(
                types.Part.from_bytes(
                    data=Path(attachment.saved_path).read_bytes(),
                    mime_type=attachment.mime_type,
                )
            )

        return contents


def truncate_text(value: str | None, *, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n\n[truncated]"


def should_inline_attachment(mime_type: str, size_bytes: int, *, max_attachment_bytes: int) -> bool:
    if size_bytes > max_attachment_bytes:
        return False
    return mime_type.startswith("image/") or mime_type == "application/pdf"


def normalize_intent_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    task_type = payload.get("task_type")

    if task_type == "unsupported" and not payload.get("reason"):
        payload["reason"] = "Prompt did not map cleanly to a supported task family"

    if task_type == "create_customer":
        normalize_customer_payload(payload)

    if task_type == "create_project":
        if "project_name" in payload and "name" not in payload:
            payload["name"] = payload.pop("project_name")
        customer = payload.get("customer")
        if isinstance(customer, dict):
            normalize_customer_payload(customer)

    if task_type == "create_invoice":
        customer = payload.get("customer")
        if isinstance(customer, dict):
            normalize_customer_payload(customer)
        elif any(key in payload for key in ("customer_name", "company_name", "name", "email")):
            customer_payload = {
                "customer_name": payload.pop("customer_name", None),
                "company_name": payload.pop("company_name", None),
                "name": payload.pop("name", None),
                "email": payload.pop("email", None),
                "invoice_email": payload.pop("invoice_email", None),
                "phone_number": payload.pop("phone_number", None),
                "phone_number_mobile": payload.pop("phone_number_mobile", None),
                "organization_number": payload.pop("organization_number", None),
                "address": payload.pop("address", None),
                "postal_code": payload.pop("postal_code", None),
                "city": payload.pop("city", None),
            }
            customer_payload = {key: value for key, value in customer_payload.items() if value is not None}
            normalize_customer_payload(customer_payload)
            if customer_payload:
                payload["customer"] = customer_payload

        if "invoice_lines" in payload and "lines" not in payload:
            payload["lines"] = payload.pop("invoice_lines")
        if "order_lines" in payload and "lines" not in payload:
            payload["lines"] = payload.pop("order_lines")

        lines = payload.get("lines")
        if isinstance(lines, list):
            for line in lines:
                if isinstance(line, dict):
                    normalize_invoice_line_payload(line)

        if "register_payment" in payload and "register_full_payment" not in payload:
            payload["register_full_payment"] = payload.pop("register_payment")

    if task_type == "create_travel_expense":
        detail_keys = {
            "departure_date",
            "return_date",
            "departure_from",
            "destination",
            "purpose",
            "is_day_trip",
            "is_foreign_travel",
            "is_compensation_from_rates",
        }
        alias_map = {
            "trip_date": "departure_date",
            "travel_date": "departure_date",
            "trip_start_date": "departure_date",
            "trip_end_date": "return_date",
            "travel_date_start": "departure_date",
            "travel_date_end": "return_date",
            "origin": "departure_from",
        }

        details = payload.get("details")
        if not isinstance(details, dict):
            details = {}

        for alias, target in alias_map.items():
            if alias in payload and target not in payload:
                payload[target] = payload[alias]

        for container_key in ("trip_dates", "travel_dates"):
            container = payload.get(container_key)
            if isinstance(container, dict):
                if "start_date" in container and "departure_date" not in payload:
                    payload["departure_date"] = container["start_date"]
                if "end_date" in container and "return_date" not in payload:
                    payload["return_date"] = container["end_date"]

        moved_any = False
        for key in list(detail_keys):
            if key in payload:
                details[key] = payload.pop(key)
                moved_any = True

        if moved_any or details:
            payload["details"] = details

        for alias in alias_map:
            payload.pop(alias, None)
        payload.pop("trip_dates", None)
        payload.pop("travel_dates", None)

    if task_type == "delete_travel_expense":
        if "travel_expense_title" in payload and "title" not in payload:
            payload["title"] = payload.pop("travel_expense_title")

    return payload


def normalize_customer_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "customer_name": "name",
        "company_name": "name",
        "customer_email": "email",
        "invoice_mail": "invoice_email",
        "invoiceEmail": "invoice_email",
        "phone": "phone_number",
        "mobile": "phone_number_mobile",
        "org_number": "organization_number",
        "orgnr": "organization_number",
        "org_no": "organization_number",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    address_line1 = first_non_none(
        payload.pop("address", None),
        payload.pop("street_address", None),
        payload.pop("address_line1", None),
    )
    postal_code = first_non_none(
        payload.pop("postal_code", None),
        payload.pop("postalCode", None),
        payload.pop("zip", None),
        payload.pop("zipcode", None),
    )
    city = first_non_none(
        payload.pop("city", None),
        payload.pop("town", None),
    )

    if any(value is not None for value in (address_line1, postal_code, city)):
        postal_address = payload.get("postal_address")
        if not isinstance(postal_address, dict):
            postal_address = {}
        if address_line1 is not None and "address_line1" not in postal_address:
            postal_address["address_line1"] = address_line1
        if postal_code is not None and "postal_code" not in postal_address:
            postal_address["postal_code"] = postal_code
        if city is not None and "city" not in postal_address:
            postal_address["city"] = city
        payload["postal_address"] = postal_address


def normalize_invoice_line_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "text": "description",
        "name": "description",
        "unit_price": "unit_price_excluding_vat_currency",
        "unitPrice": "unit_price_excluding_vat_currency",
        "price": "unit_price_excluding_vat_currency",
        "qty": "quantity",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    for key in ("unit_price_excluding_vat_currency", "unit_price_including_vat_currency"):
        value = payload.get(key)
        if isinstance(value, dict):
            amount = value.get("amount")
            if amount is not None:
                payload[key] = amount

    for key in ("product_id", "productId", "product_number", "productNumber"):
        payload.pop(key, None)


def first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None
