from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

from tripletex_agent.models import ExecutionPlan, PreparedAttachment
from tripletex_agent.planning_base import PlanningError, TaskPlanner
from tripletex_agent.task_compiler import compile_task_intent
from tripletex_agent.task_intents import SUPPORTED_TASK_INTENT_ADAPTER

_TEMPLATE = re.compile(r"{{\s*([^{}]+?)\s*}}")

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
- create_voucher
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
- If the user asks to register a supplier/vendor/leverandor/proveedor/fournisseur/fornecedor, use create_customer with is_supplier=true and is_customer=false.
- For invoices, extract customer details and invoice lines. The compiler will create customer, order, and invoice in that order.
- For invoices, default send_to_customer=false unless the prompt explicitly says to send, email, dispatch, or deliver the invoice to the customer.
- If the user asks to register payment for an invoice that is being created in the same task, use create_invoice and set register_full_payment=true.
- If the prompt refers to an already existing invoice, overdue invoice, previous invoice, existing payment, bank reconciliation, or foreign-exchange settlement on an already sent invoice, prefer task_type=unsupported unless the task also includes explicit standalone journal entries that fit create_voucher.
- Use create_invoice with is_credit_note=true when the prompt asks for a new full credit note and the reversed line items are explicitly given.
- Use create_voucher for supplier invoices when the prompt explicitly provides the accounting treatment, such as expense account, VAT, and supplier ledger posting.
- Use create_voucher for explicit journal entries, accruals, depreciation, provisions, reminder-fee postings, and other bookkeeping tasks when the prompt already specifies the debit/credit account numbers and amounts.
- create_voucher.postings must be balanced. Each posting must include account_number, entry_type (DEBIT or CREDIT), and amount.
- If a prompt mixes supported journal entries with unsupported verification or analysis, still return the supported create_voucher tasks for the explicit postings and ignore the unsupported verification-only part.
- If the prompt first requires analyzing the ledger, finding the right vouchers, reconciling bank data, or locating an overdue invoice before the postings can be known, prefer task_type=unsupported unless the final corrective postings are already explicit.
- For customer creation, flat address fields like address, postal_code, and city should be mapped into postal_address.
- Invoice lines must include description and quantity. Include either unit_price_excluding_vat_currency or unit_price_including_vat_currency when the prompt provides price information.
- The planner may return a JSON array of supported intents when the prompt clearly asks for multiple independent creates or vouchers.
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
            if isinstance(normalized_intent, list):
                intents = [
                    SUPPORTED_TASK_INTENT_ADAPTER.validate_python(item)
                    for item in normalized_intent
                ]
                return combine_execution_plans(
                    [
                        compile_task_intent(intent, allow_beta_endpoints=self.allow_beta_endpoints)
                        for intent in intents
                    ]
                )

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
    if isinstance(payload, list):
        return [normalize_intent_payload(item) for item in payload]

    if not isinstance(payload, dict):
        return payload

    task_type = payload.get("task_type")

    if task_type == "unsupported" and not payload.get("reason"):
        payload["reason"] = "Prompt did not map cleanly to a supported task family"

    if task_type == "create_employee":
        normalize_employee_payload(payload)

    if task_type == "create_customer":
        normalize_customer_payload(payload)

    if task_type == "create_project":
        if "project_name" in payload and "name" not in payload:
            payload["name"] = payload.pop("project_name")
        project_manager = payload.pop("project_manager", None)
        if isinstance(project_manager, dict):
            if "project_manager_email" not in payload and isinstance(project_manager.get("email"), str):
                payload["project_manager_email"] = project_manager["email"]
            manager_name = first_non_none(
                project_manager.get("name"),
                project_manager.get("full_name"),
            )
            if isinstance(manager_name, str):
                first_name, last_name = split_person_name(manager_name)
                if first_name and "project_manager_first_name" not in payload:
                    payload["project_manager_first_name"] = first_name
                if last_name and "project_manager_last_name" not in payload:
                    payload["project_manager_last_name"] = last_name
        if "projectManagerEmail" in payload and "project_manager_email" not in payload:
            payload["project_manager_email"] = payload.pop("projectManagerEmail")
        if "projectManagerFirstName" in payload and "project_manager_first_name" not in payload:
            payload["project_manager_first_name"] = payload.pop("projectManagerFirstName")
        if "projectManagerLastName" in payload and "project_manager_last_name" not in payload:
            payload["project_manager_last_name"] = payload.pop("projectManagerLastName")
        fixed_price_amount = payload.get("fixed_price_amount")
        if isinstance(fixed_price_amount, str):
            parsed_fixed_price_amount = parse_numeric_string(fixed_price_amount)
            if parsed_fixed_price_amount is not None:
                payload["fixed_price_amount"] = parsed_fixed_price_amount
        customer = payload.get("customer")
        if isinstance(customer, dict):
            normalize_customer_payload(customer)

    if task_type == "create_invoice":
        if "credit_note" in payload and "is_credit_note" not in payload:
            payload["is_credit_note"] = payload.pop("credit_note")

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

        payload.pop("project_name", None)
        payload.pop("currency", None)
        for key in ("currency_code", "exchange_rate", "payment_exchange_rate"):
            payload.pop(key, None)

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

        normalize_travel_expense_entry_list(payload, "per_diem_entries")
        normalize_travel_expense_entry_list(payload, "expense_entries")

        for alias in alias_map:
            payload.pop(alias, None)
        payload.pop("trip_dates", None)
        payload.pop("travel_dates", None)

    if task_type == "create_voucher":
        normalize_voucher_payload(payload)

    if task_type == "delete_travel_expense":
        if "travel_expense_title" in payload and "title" not in payload:
            payload["title"] = payload.pop("travel_expense_title")

    return payload


def normalize_customer_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "customer_name": "name",
        "company_name": "name",
        "supplier_name": "name",
        "vendor_name": "name",
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
    raw_postal_address = payload.get("postal_address")
    if isinstance(raw_postal_address, str):
        payload["postal_address"] = parse_address_string(raw_postal_address)

    raw_physical_address = payload.get("physical_address")
    if isinstance(raw_physical_address, str):
        payload["physical_address"] = parse_address_string(raw_physical_address)

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

    role = payload.get("role")
    if isinstance(role, str) and role.strip().lower() in {"supplier", "vendor"}:
        payload["is_supplier"] = True
        payload["is_customer"] = False

    supplier_hint = first_non_none(
        payload.get("is_supplier"),
        payload.get("isSupplier"),
        payload.get("supplier"),
        payload.get("vendor"),
    )
    if supplier_hint is True:
        payload["is_supplier"] = True
        payload["is_customer"] = False


def normalize_employee_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "firstName": "first_name",
        "lastName": "last_name",
        "dateOfBirth": "date_of_birth",
        "birth_date": "date_of_birth",
        "nationalIdentityNumber": "national_identity_number",
        "ssn": "national_identity_number",
        "social_security_number": "national_identity_number",
        "bankAccountNumber": "bank_account_number",
        "department": "department_name",
        "positionCode": "position_code",
        "job_code": "position_code",
        "jobCode": "position_code",
        "jobTitle": "job_title",
        "title": "job_title",
        "salary": "annual_salary",
        "salary_amount": "annual_salary",
        "employmentPercent": "employment_percentage",
        "employment_rate": "employment_percentage",
        "employmentPercentage": "employment_percentage",
        "dailyWorkingHours": "daily_working_hours",
        "working_hours_per_day": "daily_working_hours",
        "hours_per_day": "daily_working_hours",
        "standard_daily_hours": "daily_working_hours",
        "startDate": "start_date",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    for key in ("annual_salary", "employment_percentage", "daily_working_hours"):
        value = payload.get(key)
        if isinstance(value, str):
            parsed_value = parse_numeric_string(value)
            if parsed_value is not None:
                payload[key] = parsed_value

    entitlement_template = payload.get("entitlement_template")
    if isinstance(entitlement_template, str):
        normalized_entitlement_template = entitlement_template.strip().upper()
        if normalized_entitlement_template in {"EMPLOYEE", "STANDARD", "NO_ACCESS"}:
            payload.pop("entitlement_template", None)


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
        elif isinstance(value, str):
            parsed = parse_numeric_string(value)
            if parsed is not None:
                payload[key] = parsed

    quantity = payload.get("quantity")
    if isinstance(quantity, str):
        parsed_quantity = parse_numeric_string(quantity)
        if parsed_quantity is not None:
            payload["quantity"] = parsed_quantity

    vat_percent = first_non_none(
        payload.pop("vat_percent", None),
        payload.pop("vatPercent", None),
        payload.pop("vat_rate", None),
        payload.pop("vatRate", None),
    )
    if "vat_type_id" not in payload:
        vat_type_id = vat_percent_to_type_id(vat_percent)
        if vat_type_id is not None:
            payload["vat_type_id"] = vat_type_id

    for key in ("product_id", "productId", "product_number", "productNumber", "currency"):
        payload.pop(key, None)


def normalize_voucher_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "date": "voucher_date",
        "voucherDate": "voucher_date",
        "text": "description",
        "comment": "description",
        "journal_entries": "postings",
        "entries": "postings",
        "lines": "postings",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    supplier_invoice_details = payload.get("supplier_invoice_details")
    if isinstance(supplier_invoice_details, dict):
        total_amount = supplier_invoice_details.get("total_amount_including_vat")
        if isinstance(total_amount, str):
            parsed_total_amount = parse_numeric_string(total_amount)
            if parsed_total_amount is not None:
                supplier_invoice_details["total_amount_including_vat"] = parsed_total_amount

    if "amount" in payload and "postings" not in payload:
        debit_account = first_non_none(
            payload.pop("debit_account_number", None),
            payload.pop("debitAccountNumber", None),
        )
        credit_account = first_non_none(
            payload.pop("credit_account_number", None),
            payload.pop("creditAccountNumber", None),
        )
        amount = payload.get("amount")
        if debit_account is not None and credit_account is not None and amount is not None:
            payload["postings"] = [
                {
                    "account_number": debit_account,
                    "entry_type": "DEBIT",
                    "amount": amount,
                },
                {
                    "account_number": credit_account,
                    "entry_type": "CREDIT",
                    "amount": amount,
                },
            ]

    postings = payload.get("postings")
    if not isinstance(postings, list):
        return

    for posting in postings:
        if isinstance(posting, dict):
            normalize_voucher_posting_payload(posting)


def normalize_voucher_posting_payload(payload: dict[str, object]) -> None:
    if "account" in payload and "account_number" not in payload:
        account = payload.pop("account")
        if isinstance(account, dict) and "number" in account:
            payload["account_number"] = account["number"]
        elif isinstance(account, (int, str)):
            payload["account_number"] = account

    alias_map = {
        "accountNumber": "account_number",
        "account_no": "account_number",
        "accountNo": "account_number",
        "number": "account_number",
        "direction": "entry_type",
        "type": "entry_type",
        "posting_type": "entry_type",
        "vatTypeId": "vat_type_id",
        "vat_code_id": "vat_type_id",
        "department": "department_name",
        "departmentName": "department_name",
        "text": "description",
        "comment": "description",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    if "debit_amount" in payload and "amount" not in payload:
        payload["amount"] = payload.pop("debit_amount")
        payload["entry_type"] = "DEBIT"
    if "credit_amount" in payload and "amount" not in payload:
        payload["amount"] = payload.pop("credit_amount")
        payload["entry_type"] = "CREDIT"

    entry_type = payload.get("entry_type")
    if isinstance(entry_type, str):
        normalized = entry_type.strip().upper()
        if normalized in {"DEBIT", "DR"}:
            payload["entry_type"] = "DEBIT"
        elif normalized in {"CREDIT", "CR"}:
            payload["entry_type"] = "CREDIT"

    account_number = payload.get("account_number")
    if isinstance(account_number, str):
        parsed_account_number = parse_numeric_string(account_number)
        if parsed_account_number is not None:
            payload["account_number"] = int(parsed_account_number)

    for key in ("amount", "vat_type_id"):
        value = payload.get(key)
        if isinstance(value, str):
            parsed_value = parse_numeric_string(value)
            if parsed_value is None:
                continue
            payload[key] = parsed_value if key == "amount" else int(parsed_value)

    department_name = payload.get("department_name")
    if isinstance(department_name, dict):
        payload["department_name"] = first_non_none(
            department_name.get("name"),
            department_name.get("department_name"),
        )


def first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def parse_numeric_string(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"[^\d,.\-]", "", cleaned)
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_address_string(value: str) -> dict[str, object]:
    cleaned = value.strip()
    if not cleaned:
        return {"address_line1": value}

    match = re.match(r"^(?P<address>.+?),\s*(?P<postal>\d{4})\s+(?P<city>.+)$", cleaned)
    if match:
        return {
            "address_line1": match.group("address").strip(),
            "postal_code": match.group("postal").strip(),
            "city": match.group("city").strip(),
        }

    return {"address_line1": cleaned}


def split_person_name(value: str) -> tuple[str | None, str | None]:
    parts = [part for part in value.strip().split() if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def vat_percent_to_type_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = parse_numeric_string(value)
        if parsed is None:
            return None
        value = parsed
    if not isinstance(value, (int, float)):
        return None
    rounded = round(float(value), 2)
    mapping = {
        25.0: 3,
        15.0: 31,
        12.0: 32,
        11.11: 311,
        0.0: 5,
    }
    return mapping.get(rounded)


def normalize_travel_expense_entry_list(payload: dict[str, object], key: str) -> None:
    entries = payload.get(key)
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return

    normalized_entries: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized_entry = dict(entry)
        for amount_key in (
            "amount_per_day_currency",
            "amount_currency",
            "amount",
            "number_of_days",
        ):
            value = normalized_entry.get(amount_key)
            if isinstance(value, str):
                parsed_value = parse_numeric_string(value)
                if parsed_value is not None:
                    normalized_entry[amount_key] = parsed_value
        normalized_entries.append(normalized_entry)

    payload[key] = normalized_entries


def combine_execution_plans(plans: list[ExecutionPlan]) -> ExecutionPlan:
    if not plans:
        raise PlanningError("Gemini planner returned an empty task list")

    combined_actions = []
    combined_notes: list[str] = []
    combined_goals: list[str] = []
    for index, plan in enumerate(plans, start=1):
        prefix = f"batch{index}_"
        renamed_plan = rename_execution_plan(plan, prefix)
        combined_actions.extend(renamed_plan.actions)
        combined_notes.extend(renamed_plan.verification_notes)
        combined_goals.append(plan.goal)

    return ExecutionPlan(
        goal="; ".join(combined_goals),
        actions=combined_actions,
        verification_notes=dedupe_preserve_order(combined_notes),
    )


def rename_execution_plan(plan: ExecutionPlan, prefix: str) -> ExecutionPlan:
    mapping: dict[str, str] = {}
    for action in plan.actions:
        mapping[action.id] = f"{prefix}{action.id}"
        if action.save_as:
            mapping[action.save_as] = f"{prefix}{action.save_as}"

    renamed_actions = []
    for action in plan.actions:
        renamed_actions.append(
            action.model_copy(
                update={
                    "id": mapping[action.id],
                    "path": rename_templates(action.path, mapping),
                    "params": rename_templates(action.params, mapping),
                    "body": rename_templates(action.body, mapping),
                    "save_as": mapping.get(action.save_as) if action.save_as else None,
                }
            )
        )

    return plan.model_copy(update={"actions": renamed_actions})


def rename_templates(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, str):
        def replace_match(match: re.Match[str]) -> str:
            reference = match.group(1)
            parts = reference.split(".", 1)
            head = mapping.get(parts[0], parts[0])
            new_reference = head if len(parts) == 1 else f"{head}.{parts[1]}"
            return "{{" + new_reference + "}}"

        return _TEMPLATE.sub(replace_match, value)

    if isinstance(value, list):
        return [rename_templates(item, mapping) for item in value]

    if isinstance(value, dict):
        return {key: rename_templates(item, mapping) for key, item in value.items()}

    return value


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
