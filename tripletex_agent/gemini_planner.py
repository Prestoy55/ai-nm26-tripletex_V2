from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

from tripletex_agent.models import ExecutionPlan, PreparedAttachment
from tripletex_agent.planning_base import PlanningError, TaskPlanner
from tripletex_agent.task_compiler import compile_task_intent
from tripletex_agent.task_intents import (
    SUPPORTED_TASK_INTENT_ADAPTER,
    CreateInvoiceIntent,
    CreateProjectIntent,
    CreateTravelExpenseIntent,
    CreateVoucherIntent,
    InvoiceCustomerIntent,
    VoucherPostingIntent,
)

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
- If the prompt mixes supported project creation with unsupported lifecycle steps like time logging, supplier costs without explicit journal postings, or invoice generation without concrete line items, still return create_project and ignore the unsupported parts.
- If the user asks to register a supplier/vendor/leverandor/proveedor/fournisseur/fornecedor, use create_customer with is_supplier=true and is_customer=false.
- For invoices, extract customer details and invoice lines. The compiler will create customer, order, and invoice in that order.
- For invoices, default send_to_customer=false unless the prompt explicitly says to send, email, dispatch, or deliver the invoice to the customer.
- If the user asks to register payment for an invoice that is being created in the same task, use create_invoice and set register_full_payment=true.
- Competition submissions start from a fresh account. If the prompt describes an existing or outstanding customer invoice but also provides enough explicit customer, line, and amount details to synthesize the requested end state in a fresh account, you may still use create_invoice. For example, a full-payment task can become create_invoice with register_full_payment=true, while a returned-payment task can become create_invoice without payment so the invoice remains outstanding.
- If the prompt refers to an already existing invoice, overdue invoice, previous invoice, existing payment, bank reconciliation, or foreign-exchange settlement on an already sent invoice and the missing invoice state cannot be reconstructed from the prompt, prefer task_type=unsupported unless the task also includes explicit standalone journal entries that fit create_voucher.
- Use create_invoice with is_credit_note=true when the prompt asks for a new full credit note and the reversed line items are explicitly given.
- Use create_voucher for supplier invoices when the prompt explicitly provides the accounting treatment, such as expense account, VAT, and supplier ledger posting.
- For supplier invoices, unless the prompt explicitly gives another liability account, use account 2400 for the supplier ledger credit posting.
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
        deterministic_intent = build_deterministic_intent(prompt)
        if deterministic_intent is not None:
            return compile_task_intent(
                deterministic_intent,
                allow_beta_endpoints=self.allow_beta_endpoints,
            )

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
        normalized_intent = enrich_intent_payload_from_prompt(prompt, normalized_intent)

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

        if getattr(intent, "task_type", None) == "unsupported":
            fallback_intent = build_explicit_voucher_fallback(prompt)
            if fallback_intent is not None:
                return compile_task_intent(
                    fallback_intent,
                    allow_beta_endpoints=self.allow_beta_endpoints,
                )
            invoice_fallback_intent = build_fresh_invoice_state_fallback(prompt)
            if invoice_fallback_intent is not None:
                return compile_task_intent(
                    invoice_fallback_intent,
                    allow_beta_endpoints=self.allow_beta_endpoints,
                )
            project_fallback_intent = build_project_fallback(prompt)
            if project_fallback_intent is not None:
                return compile_task_intent(
                    project_fallback_intent,
                    allow_beta_endpoints=self.allow_beta_endpoints,
                )

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
        normalized_items = [normalize_intent_payload(item) for item in payload]
        normalize_intent_payload_list(normalized_items)
        return normalized_items

    if not isinstance(payload, dict):
        return payload

    task_type = payload.get("task_type")

    if task_type == "unsupported" and not payload.get("reason"):
        payload["reason"] = "Prompt did not map cleanly to a supported task family"

    if task_type == "create_employee":
        normalize_employee_payload(payload)

    if task_type == "create_customer":
        normalize_customer_payload(payload)

    if task_type == "create_product":
        normalize_product_payload(payload)

    if task_type == "create_project":
        if "project_name" in payload and "name" not in payload:
            payload["name"] = payload.pop("project_name")
        if "fixed_price_amount_currency" in payload and "fixed_price_amount" not in payload:
            payload["fixed_price_amount"] = payload.pop("fixed_price_amount_currency")
        if "fixedPriceAmountCurrency" in payload and "fixed_price_amount" not in payload:
            payload["fixed_price_amount"] = payload.pop("fixedPriceAmountCurrency")
        project_manager = payload.pop("project_manager", None)
        if project_manager is None:
            project_manager = payload.pop("project_leader", None)
        if project_manager is None:
            project_manager = payload.pop("projectLeader", None)
        if isinstance(project_manager, dict):
            if "project_manager_email" not in payload and isinstance(project_manager.get("email"), str):
                payload["project_manager_email"] = project_manager["email"]
            manager_name = first_non_none(
                project_manager.get("name"),
                project_manager.get("full_name"),
            )
            if not isinstance(manager_name, str):
                first_name = project_manager.get("first_name")
                last_name = project_manager.get("last_name")
                if isinstance(first_name, str) and "project_manager_first_name" not in payload:
                    payload["project_manager_first_name"] = first_name
                if isinstance(last_name, str) and "project_manager_last_name" not in payload:
                    payload["project_manager_last_name"] = last_name
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
        for unsupported_budget_key in (
            "budget_amount",
            "budgetAmount",
            "budget",
            "project_budget",
            "projectBudget",
            "budget_amount_currency",
            "budgetAmountCurrency",
        ):
            payload.pop(unsupported_budget_key, None)
        customer = payload.get("customer")
        if isinstance(customer, dict):
            normalize_customer_payload(customer)

    if task_type == "create_invoice":
        if "credit_note" in payload and "is_credit_note" not in payload:
            payload["is_credit_note"] = payload.pop("credit_note")

        if "customer_details" in payload and "customer" not in payload:
            payload["customer"] = payload.pop("customer_details")
        if "customerDetails" in payload and "customer" not in payload:
            payload["customer"] = payload.pop("customerDetails")

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
        if "due_date" in payload and "invoice_due_date" not in payload:
            payload["invoice_due_date"] = payload.pop("due_date")
        if "invoiceDueDate" in payload and "invoice_due_date" not in payload:
            payload["invoice_due_date"] = payload.pop("invoiceDueDate")

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

        expenses = payload.pop("expenses", None)
        if expenses is None:
            expenses = payload.pop("travel_expense_lines", None)
        if isinstance(expenses, dict):
            expenses = [expenses]

        daily_allowance_days = payload.pop("daily_allowance_days", None)
        if daily_allowance_days is None:
            daily_allowance_days = payload.pop("per_diem_days", None)
        daily_allowance_amount = payload.pop("daily_allowance_amount", None)
        if daily_allowance_amount is None:
            daily_allowance_amount = payload.pop("per_diem_amount", None)

        if daily_allowance_days is not None or daily_allowance_amount is not None:
            per_diem_entries = payload.get("per_diem_entries")
            if not isinstance(per_diem_entries, list):
                per_diem_entries = []
            per_diem_entries.append(
                prune_none_dict(
                    {
                        "number_of_days": daily_allowance_days,
                        "amount_per_day_currency": daily_allowance_amount,
                    }
                )
            )
            payload["per_diem_entries"] = per_diem_entries

        if isinstance(expenses, list):
            per_diem_entries = payload.get("per_diem_entries")
            if not isinstance(per_diem_entries, list):
                per_diem_entries = []
            expense_entries = payload.get("expense_entries")
            if not isinstance(expense_entries, list):
                expense_entries = []

            for entry in expenses:
                if not isinstance(entry, dict):
                    continue
                entry_type = str(first_non_none(entry.get("type"), entry.get("expense_type")) or "").strip().lower()
                amount = entry.get("amount")
                description = entry.get("description")
                currency = entry.get("currency")
                if entry_type in {"daily_allowance", "per_diem", "diem", "diet"}:
                    number_of_days = None
                    if isinstance(description, str):
                        days_match = re.search(r"(\d+)\s+days?", description, flags=re.IGNORECASE)
                        if days_match:
                            number_of_days = int(days_match.group(1))
                    per_diem_entries.append(
                        prune_none_dict(
                            {
                                "number_of_days": number_of_days,
                                "amount_per_day_currency": amount if number_of_days in (None, 0) else divide_amount(amount, number_of_days),
                                "amount_currency": amount,
                                "currency": currency,
                                "description": description,
                            }
                        )
                    )
                else:
                    expense_entries.append(
                        prune_none_dict(
                            {
                                "description": description,
                                "amount_currency": amount,
                                "currency": currency,
                            }
                        )
                    )

            if per_diem_entries:
                payload["per_diem_entries"] = per_diem_entries
            if expense_entries:
                payload["expense_entries"] = expense_entries

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
    elif isinstance(raw_postal_address, dict):
        normalize_address_payload(raw_postal_address)

    raw_physical_address = payload.get("physical_address")
    if isinstance(raw_physical_address, str):
        payload["physical_address"] = parse_address_string(raw_physical_address)
    elif isinstance(raw_physical_address, dict):
        normalize_address_payload(raw_physical_address)

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
        "occupation_code": "position_code",
        "occupationCode": "position_code",
        "position": "job_title",
        "jobTitle": "job_title",
        "title": "job_title",
        "salary": "annual_salary",
        "salary_amount": "annual_salary",
        "employmentPercent": "employment_percentage",
        "employment_rate": "employment_percentage",
        "employmentPercentage": "employment_percentage",
        "fte_percentage": "employment_percentage",
        "position_percentage": "employment_percentage",
        "dailyWorkingHours": "daily_working_hours",
        "daily_work_hours": "daily_working_hours",
        "working_hours_per_day": "daily_working_hours",
        "hours_per_day": "daily_working_hours",
        "standard_daily_hours": "daily_working_hours",
        "startDate": "start_date",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    department_name = payload.get("department_name")
    if isinstance(department_name, dict):
        payload["department_name"] = first_non_none(
            department_name.get("name"),
            department_name.get("department_name"),
        )

    for key in ("annual_salary", "employment_percentage", "daily_working_hours"):
        value = payload.get(key)
        if isinstance(value, str):
            parsed_value = parse_numeric_string(value)
            if parsed_value is not None:
                payload[key] = parsed_value

    payload.pop("employment_type", None)

    entitlement_template = payload.get("entitlement_template")
    if isinstance(entitlement_template, str):
        normalized_entitlement_template = entitlement_template.strip().upper()
        if normalized_entitlement_template in {"EMPLOYEE", "STANDARD", "NO_ACCESS"}:
            payload.pop("entitlement_template", None)


def normalize_product_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "product_name": "name",
        "productName": "name",
        "product_number": "number",
        "productNumber": "number",
        "unit_price_excluding_vat_currency": "price_excluding_vat_currency",
        "unitPriceExcludingVatCurrency": "price_excluding_vat_currency",
        "unit_price_including_vat_currency": "price_including_vat_currency",
        "unitPriceIncludingVatCurrency": "price_including_vat_currency",
        "cost_price_excluding_vat_currency": "cost_excluding_vat_currency",
        "costPriceExcludingVatCurrency": "cost_excluding_vat_currency",
        "vat_rate_percent": "vat_rate_percent",
        "vatPercent": "vat_rate_percent",
        "vat_percent": "vat_rate_percent",
    }

    for alias, target in alias_map.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    for key in (
        "price_excluding_vat_currency",
        "price_including_vat_currency",
        "cost_excluding_vat_currency",
    ):
        value = payload.get(key)
        if isinstance(value, str):
            parsed_value = parse_numeric_string(value)
            if parsed_value is not None:
                payload[key] = parsed_value

    vat_percent = payload.pop("vat_rate_percent", None)
    if "vat_type_id" not in payload:
        vat_type_id = first_non_none(
            vat_percent_to_type_id(vat_percent),
            vat_name_to_type_id(
                first_non_none(
                    payload.pop("vat_type", None),
                    payload.pop("vatType", None),
                )
            ),
        )
        if vat_type_id is not None:
            payload["vat_type_id"] = vat_type_id

    payload.pop("currency", None)


def normalize_invoice_line_payload(payload: dict[str, object]) -> None:
    alias_map = {
        "text": "description",
        "name": "description",
        "unit_price": "unit_price_excluding_vat_currency",
        "unitPrice": "unit_price_excluding_vat_currency",
        "price": "unit_price_excluding_vat_currency",
        "qty": "quantity",
        "vat_rate_percent": "vat_percent",
        "vatRatePercent": "vat_percent",
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
        vat_type_id = first_non_none(
            vat_percent_to_type_id(vat_percent),
            vat_name_to_type_id(
                first_non_none(
                    payload.pop("vat_type", None),
                    payload.pop("vatType", None),
                )
            ),
        )
        if vat_type_id is not None:
            payload["vat_type_id"] = vat_type_id

    payload.pop("currency", None)

    for key in (
        "product_id",
        "productId",
        "product_number",
        "productNumber",
        "currency",
        "unit_price_currency",
        "unitPriceCurrency",
    ):
        payload.pop(key, None)


def normalize_intent_payload_list(payloads: list[object]) -> None:
    customer_names_by_org: dict[str, str] = {}

    for item in payloads:
        if not isinstance(item, dict):
            continue
        customer = item.get("customer")
        if not isinstance(customer, dict):
            continue
        organization_number = customer.get("organization_number")
        name = customer.get("name")
        if isinstance(organization_number, str) and isinstance(name, str) and name.strip():
            customer_names_by_org[organization_number.strip()] = name

    if not customer_names_by_org:
        return

    for item in payloads:
        if not isinstance(item, dict):
            continue
        customer = item.get("customer")
        if not isinstance(customer, dict):
            continue
        if isinstance(customer.get("name"), str) and customer["name"].strip():
            continue
        organization_number = customer.get("organization_number")
        if not isinstance(organization_number, str):
            continue
        inferred_name = customer_names_by_org.get(organization_number.strip())
        if inferred_name:
            customer["name"] = inferred_name


def enrich_intent_payload_from_prompt(prompt: str, payload: object) -> object:
    if isinstance(payload, list):
        return [enrich_intent_payload_from_prompt(prompt, item) for item in payload]
    if not isinstance(payload, dict):
        return payload
    if payload.get("task_type") != "create_voucher":
        return payload

    if not is_payroll_prompt(prompt):
        return payload

    employee = extract_employee_identity_from_prompt(prompt)
    if employee is None:
        return payload

    if employee.get("first_name") and "employee_first_name" not in payload:
        payload["employee_first_name"] = employee["first_name"]
    if employee.get("last_name") and "employee_last_name" not in payload:
        payload["employee_last_name"] = employee["last_name"]
    if employee.get("email") and "employee_email" not in payload:
        payload["employee_email"] = employee["email"]
    return payload


def is_payroll_prompt(prompt: str) -> bool:
    lowered = normalize_text_for_match(prompt)
    return any(
        token in lowered
        for token in (
            "payroll",
            "salary",
            "bonus",
            "gehaltsabrechnung",
            "grundgehalt",
            "gehalt",
            "lonn",
            "lønn",
            "salario",
            "folha",
        )
    )


def extract_employee_identity_from_prompt(prompt: str) -> dict[str, str] | None:
    email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", prompt, flags=re.IGNORECASE)
    first_name: str | None = None
    last_name: str | None = None
    if email_match:
        email_value = email_match.group(1)
        local_part = email_value.split("@", 1)[0]
        local_parts = [part for part in re.split(r"[._\-]+", local_part) if part]
        if len(local_parts) >= 2:
            first_name = local_parts[0].capitalize()
            last_name = " ".join(part.capitalize() for part in local_parts[1:])
        leading_text = prompt[: email_match.start()]
        name_match = re.search(r"(?P<name>[^()]{2,120})\s*\([^()]*$", leading_text)
        if name_match and (not first_name or not last_name):
            raw_name = name_match.group("name").strip(" ,:")
            raw_name = re.sub(
                r"^.*\b(?:for|fur|pour|para|til|for denne maned durch|for this month through|for this month|named|employee named|mitarbeiter namens)\s+",
                "",
                raw_name,
                flags=re.IGNORECASE,
            ).strip(" ,:")
            first_name, last_name = split_person_name(raw_name)

    if not first_name or not last_name:
        fallback_match = re.search(
            r"(?:for|fur|pour|para|til|named|employee named|mitarbeiter namens)\s+([^,().]{3,120})",
            prompt,
            flags=re.IGNORECASE,
        )
        if fallback_match:
            first_name, last_name = split_person_name(fallback_match.group(1).strip(" ,:"))

    if not first_name or not last_name:
        return None

    result = {"first_name": first_name, "last_name": last_name}
    if email_match:
        result["email"] = email_match.group(1)
    return result


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
    if not isinstance(supplier_invoice_details, dict):
        supplier_invoice_details = {}

    supplier_aliases = {
        "supplier_name": "supplier_name",
        "vendor_name": "supplier_name",
        "organization_number": "organization_number",
        "org_number": "organization_number",
        "supplier_organization_number": "organization_number",
        "supplier_org_number": "organization_number",
        "supplier_orgnr": "organization_number",
        "supplier_address": "supplier_address",
        "invoice_number": "invoice_number",
        "external_id": "invoice_number",
        "invoice_date": "invoice_date",
        "due_date": "due_date",
        "total_amount_including_vat": "total_amount_including_vat",
        "currency": "currency",
    }
    for alias, target in supplier_aliases.items():
        if alias in payload and target not in supplier_invoice_details:
            supplier_invoice_details[target] = payload.pop(alias)

    supplier = payload.pop("supplier", None)
    if isinstance(supplier, dict):
        supplier_name = supplier.get("name")
        if isinstance(supplier_name, str) and "supplier_name" not in supplier_invoice_details:
            supplier_invoice_details["supplier_name"] = supplier_name
        supplier_org = first_non_none(
            supplier.get("organization_number"),
            supplier.get("org_number"),
            supplier.get("orgnr"),
        )
        if isinstance(supplier_org, str) and "organization_number" not in supplier_invoice_details:
            supplier_invoice_details["organization_number"] = supplier_org

    employee = payload.pop("employee", None)
    if isinstance(employee, dict):
        employee_email = employee.get("email")
        if isinstance(employee_email, str) and "employee_email" not in payload:
            payload["employee_email"] = employee_email
        employee_name = first_non_none(
            employee.get("name"),
            employee.get("full_name"),
        )
        if isinstance(employee_name, str):
            first_name, last_name = split_person_name(employee_name)
            if first_name and "employee_first_name" not in payload:
                payload["employee_first_name"] = first_name
            if last_name and "employee_last_name" not in payload:
                payload["employee_last_name"] = last_name

    employee_aliases = {
        "employee_email": "employee_email",
        "employeeEmail": "employee_email",
        "employee_first_name": "employee_first_name",
        "employeeFirstName": "employee_first_name",
        "employee_last_name": "employee_last_name",
        "employeeLastName": "employee_last_name",
    }
    for alias, target in employee_aliases.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

    description = payload.get("description")
    if isinstance(description, str):
        supplier_from_description = re.search(
            r"Supplier invoice\s+[A-Z0-9][A-Z0-9\-\/]*\s+from\s+(?P<name>.+?)(?:\s+for\b|$)",
            description,
            flags=re.IGNORECASE,
        )
        if supplier_from_description and "supplier_name" not in supplier_invoice_details:
            supplier_name = supplier_from_description.group("name").strip(" ,:")
            if supplier_name:
                supplier_invoice_details["supplier_name"] = supplier_name

    payload.pop("voucher_type", None)

    if isinstance(supplier_invoice_details, dict):
        total_amount = supplier_invoice_details.get("total_amount_including_vat")
        if isinstance(total_amount, str):
            parsed_total_amount = parse_numeric_string(total_amount)
            if parsed_total_amount is not None:
                supplier_invoice_details["total_amount_including_vat"] = parsed_total_amount
        if supplier_invoice_details:
            payload["supplier_invoice_details"] = supplier_invoice_details

    customer_aliases = {
        "customer_name": "customer_name",
        "client_name": "customer_name",
        "customer_organization_number": "customer_organization_number",
        "customer_org_number": "customer_organization_number",
        "customer_orgnr": "customer_organization_number",
    }
    for alias, target in customer_aliases.items():
        if alias in payload and target not in payload:
            payload[target] = payload.pop(alias)

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
        if not isinstance(posting, dict):
            continue
        posting_customer_name = first_non_none(
            posting.get("customer_name"),
            posting.get("client_name"),
        )
        if isinstance(posting_customer_name, str) and "customer_name" not in payload:
            payload["customer_name"] = posting_customer_name
        posting_customer_org = first_non_none(
            posting.get("customer_organization_number"),
            posting.get("customer_org_number"),
            posting.get("customer_orgnr"),
        )
        if isinstance(posting_customer_org, str) and "customer_organization_number" not in payload:
            payload["customer_organization_number"] = posting_customer_org

    for posting in postings:
        if isinstance(posting, dict):
            normalize_voucher_posting_payload(posting)

    if supplier_invoice_details:
        has_supplier_ledger_posting = False
        credit_postings: list[dict[str, object]] = []
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            account_number = posting.get("account_number")
            if isinstance(account_number, int) and 2400 <= account_number <= 2499:
                has_supplier_ledger_posting = True
            if posting.get("entry_type") == "CREDIT":
                credit_postings.append(posting)
        if not has_supplier_ledger_posting and len(credit_postings) == 1:
            credit_postings[0]["account_number"] = 2400


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

    for key in (
        "customer_name",
        "client_name",
        "customer_organization_number",
        "customer_org_number",
        "customer_orgnr",
    ):
        payload.pop(key, None)


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


def normalize_address_payload(payload: dict[str, object]) -> None:
    if "address" in payload and "address_line1" not in payload:
        payload["address_line1"] = payload.pop("address")
    if "addressLine1" in payload and "address_line1" not in payload:
        payload["address_line1"] = payload.pop("addressLine1")

    country = payload.pop("country", None)
    if "country_id" not in payload:
        if isinstance(country, dict):
            country_id = first_non_none(country.get("id"), country.get("country_id"))
            if isinstance(country_id, (int, float)):
                payload["country_id"] = int(country_id)
            elif isinstance(country_id, str):
                parsed_country_id = parse_numeric_string(country_id)
                if parsed_country_id is not None:
                    payload["country_id"] = int(parsed_country_id)
        elif isinstance(country, (int, float)):
            payload["country_id"] = int(country)
        elif isinstance(country, str):
            parsed_country_id = parse_numeric_string(country)
            if parsed_country_id is not None:
                payload["country_id"] = int(parsed_country_id)


def split_person_name(value: str) -> tuple[str | None, str | None]:
    parts = [part for part in value.strip().split() if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def normalize_text_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return without_accents.lower()


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


def vat_name_to_type_id(value: object) -> int | None:
    if not isinstance(value, str):
        return None

    normalized = re.sub(r"[^A-Z_]", "_", value.strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    mapping = {
        "HIGH": 3,
        "STANDARD": 3,
        "FULL": 3,
        "MEDIUM": 31,
        "LOW": 31,
        "REDUCED": 31,
        "EXEMPT": 5,
        "ZERO": 5,
        "ZERO_RATED": 5,
        "VAT_FREE": 5,
        "NO_VAT": 5,
        "NONE": 5,
    }
    return mapping.get(normalized)


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


def prune_none_dict(value: dict[str, object | None]) -> dict[str, object]:
    return {key: item for key, item in value.items() if item is not None}


def divide_amount(value: object, divisor: int) -> float | None:
    if divisor <= 0:
        return None
    if isinstance(value, str):
        value = parse_numeric_string(value)
    if isinstance(value, (int, float)):
        return round(float(value) / divisor, 2)
    return None


def build_deterministic_intent(prompt: str) -> object | None:
    for builder in (
        build_supplier_invoice_fallback,
        build_travel_expense_fallback,
        build_hours_invoice_fallback,
        build_credit_note_fallback,
        build_fresh_invoice_state_fallback,
    ):
        intent = builder(prompt)
        if intent is not None:
            return intent
    return None


def build_explicit_voucher_fallback(prompt: str) -> CreateVoucherIntent | None:
    debit_match = re.search(
        r"(?:debit(?:o|e)?|debet(?:er)?|debiter(?:a|e)?|debito)[^0-9]{0,40}\(?\s*(\d{4})\s*\)?",
        prompt,
        flags=re.IGNORECASE,
    )
    credit_match = re.search(
        r"(?:credit(?:o)?|kredit(?:er)?|credito)[^0-9]{0,40}\(?\s*(\d{4})\s*\)?",
        prompt,
        flags=re.IGNORECASE,
    )
    if not debit_match or not credit_match:
        return None

    amount = extract_explicit_voucher_amount(prompt)
    if amount is None:
        return None

    description = infer_voucher_description(prompt, amount)
    return CreateVoucherIntent(
        task_type="create_voucher",
        voucher_date=date.today().isoformat(),
        description=description,
        postings=[
            VoucherPostingIntent(
                account_number=int(debit_match.group(1)),
                entry_type="DEBIT",
                amount=amount,
            ),
            VoucherPostingIntent(
                account_number=int(credit_match.group(1)),
                entry_type="CREDIT",
                amount=amount,
            ),
        ],
    )


def extract_explicit_voucher_amount(prompt: str) -> float | None:
    prioritized_patterns = [
        r"(?:reminder|recordatorio|påminn(?:else|ingsgebyr)|rappel|mahn(?:gebühr|ung)|fee)[^0-9]{0,40}(\d[\d\s.,]*)\s*(?:NOK|kr)\b",
        r"(\d[\d\s.,]*)\s*(?:NOK|kr)\b[^.]{0,80}(?:debit(?:o|e)?|debet|credit(?:o)?|kredit)",
    ]
    for pattern in prioritized_patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        parsed = parse_numeric_string(match.group(1))
        if parsed is not None:
            return parsed

    amounts = [
        parse_numeric_string(match.group(1))
        for match in re.finditer(r"(\d[\d\s.,]*)\s*(?:NOK|kr)\b", prompt, flags=re.IGNORECASE)
    ]
    amounts = [amount for amount in amounts if amount is not None]
    if len(amounts) == 1:
        return amounts[0]
    return None


def infer_voucher_description(prompt: str, amount: float) -> str:
    lowered = prompt.lower()
    if any(token in lowered for token in ("reminder", "recordatorio", "påminn", "rappel", "mahn")):
        return f"Reminder fee posting {amount:.2f}"
    return f"Journal entry {amount:.2f}"


def build_project_fallback(prompt: str) -> CreateProjectIntent | None:
    if "project" not in prompt.lower() and "prosjekt" not in prompt.lower():
        return None

    project_name_match = re.search(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]", prompt)
    if not project_name_match:
        return None
    project_name = project_name_match.group(1).strip()
    if not project_name:
        return None

    org_match = re.search(
        r"\((?P<customer>[^()]+?),\s*(?:org(?:\.|anization)?(?:\s*no\.?|\s*nr\.?)?)\s*(?P<org>\d{9})\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not org_match:
        return None

    customer = InvoiceCustomerIntent(
        name=org_match.group("customer").strip(),
        organization_number=org_match.group("org").strip(),
    )

    email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", prompt, flags=re.IGNORECASE)
    manager_first_name: str | None = None
    manager_last_name: str | None = None
    if email_match:
        leading_text = prompt[: email_match.start()]
        name_match = re.search(r"([A-ZÆØÅÀ-ÿ][A-Za-zÆØÅæøåÀ-ÿ'’\-]+(?:\s+[A-ZÆØÅÀ-ÿ][A-Za-zÆØÅæøåÀ-ÿ'’\-]+)+)\s*\([^()]*$", leading_text)
        if name_match:
            manager_first_name, manager_last_name = split_person_name(name_match.group(1))

    return CreateProjectIntent(
        task_type="create_project",
        name=project_name,
        customer=customer,
        project_manager_email=email_match.group(1) if email_match else None,
        project_manager_first_name=manager_first_name,
        project_manager_last_name=manager_last_name,
    )


def build_supplier_invoice_fallback(prompt: str) -> CreateVoucherIntent | None:
    lowered = normalize_text_for_match(prompt)
    if not any(
        token in lowered
        for token in (
            "supplier invoice",
            "leverandørfaktura",
            "leverandorfaktura",
            "factura del proveedor",
            "facture fournisseur",
            "fatura do fornecedor",
        )
    ) and "from the supplier" not in lowered:
        return None

    supplier_match = re.search(
        (
            r"(?:from\s+the\s+supplier|from\s+supplier|fra\s+leverand[øo]ren|"
            r"fra\s+leverand[øo]r|del\s+proveedor|du\s+fournisseur|do\s+fornecedor)\s+"
            r"(?P<name>[^()]+?)\s*\((?:[^)]*?)(?:org(?:\.|anization)?(?:\s*no\.?|\s*nr\.?)?)\s*"
            r"(?P<org>\d{9})\)"
        ),
        prompt,
        flags=re.IGNORECASE,
    )
    if not supplier_match:
        return None

    invoice_number_match = re.search(
        r"(?:invoice|faktura|facture|factura|fatura)\s+([A-Z0-9][A-Z0-9\-\/]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    account_match = re.search(
        r"(?:account|konto|compte|cuenta)\s*\(?\s*(\d{4})\s*\)?",
        prompt,
        flags=re.IGNORECASE,
    )
    total_amount_match = re.search(
        (
            r"for\s+(\d[\d\s.,]*)\s*(?:NOK|kr)\b[^.]{0,60}"
            r"(?:including\s+vat|inkl(?:usiv)?\s+mva|incl(?:uant|usive)?\s+tva|com\s+iva)"
        ),
        prompt,
        flags=re.IGNORECASE,
    )
    vat_match = re.search(r"(\d{1,2})\s*%", prompt)

    if not invoice_number_match or not account_match or not total_amount_match or not vat_match:
        return None

    total_amount = parse_numeric_string(total_amount_match.group(1))
    vat_percent = parse_numeric_string(vat_match.group(1))
    if total_amount is None or vat_percent is None or vat_percent < 0:
        return None

    net_amount = round(total_amount / (1 + vat_percent / 100), 2)
    vat_amount = round(total_amount - net_amount, 2)

    supplier_name = supplier_match.group("name").strip(" ,:")
    if not supplier_name:
        return None

    return CreateVoucherIntent(
        task_type="create_voucher",
        voucher_date=date.today().isoformat(),
        description=f"Supplier invoice {invoice_number_match.group(1)} from {supplier_name}",
        supplier_invoice_details={
            "supplier_name": supplier_name,
            "organization_number": supplier_match.group("org").strip(),
            "invoice_number": invoice_number_match.group(1),
            "total_amount_including_vat": total_amount,
            "currency": "NOK",
        },
        postings=[
            VoucherPostingIntent(
                account_number=int(account_match.group(1)),
                entry_type="DEBIT",
                amount=net_amount,
            ),
            VoucherPostingIntent(
                account_number=2710,
                entry_type="DEBIT",
                amount=vat_amount,
            ),
            VoucherPostingIntent(
                account_number=2400,
                entry_type="CREDIT",
                amount=total_amount,
            ),
        ],
    )


def build_travel_expense_fallback(prompt: str) -> CreateTravelExpenseIntent | None:
    lowered = normalize_text_for_match(prompt)
    has_travel_markers = any(
        token in lowered
        for token in (
            "travel expense",
            "reisekost",
            "despesa de viagem",
            "frais de d",
            "reisekosten",
            "indemnit",
            "daily allowance",
            "per diem",
        )
    )
    has_expense_markers = re.search(
        r"(?:expenses?:|d.penses|utgifter|despesas)",
        lowered,
        flags=re.IGNORECASE,
    )
    if not has_travel_markers and not has_expense_markers:
        return None

    email_match = re.search(r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})", prompt, flags=re.IGNORECASE)
    if not email_match:
        return None

    leading_text = prompt[: email_match.start()]
    employee_name_match = re.search(
        r"([A-ZÆØÅÀ-ÿ][A-Za-zÆØÅæøåÀ-ÿ'’\-]+(?:\s+[A-ZÆØÅÀ-ÿ][A-Za-zÆØÅæøåÀ-ÿ'’\-]+)+)\s*\($",
        leading_text,
    )
    if not employee_name_match:
        return None
    first_name, last_name = split_person_name(employee_name_match.group(1))
    if not first_name or not last_name:
        return None

    title = extract_invoice_line_description(prompt)
    if not title:
        return None

    day_match = re.search(
        r"(\d+)\s+(?:days?|dager?|jours?)\b[^.]{0,60}?(\d[\d\s.,]*)\s*(?:NOK|kr)\s*(?:/day|per day|pr dag|par jour)?",
        lowered,
        flags=re.IGNORECASE,
    )
    per_diem_entries: list[dict[str, object]] = []
    if day_match:
        number_of_days = int(day_match.group(1))
        amount_per_day = parse_numeric_string(day_match.group(2))
        if amount_per_day is not None:
            per_diem_entries.append(
                {
                    "number_of_days": number_of_days,
                    "amount_per_day_currency": amount_per_day,
                }
            )

    expense_entries: list[dict[str, object]] = []
    expense_section_match = re.search(
        r"(?:expenses?|d.penses?|utgifter|despesas)\s*:\s*",
        lowered,
        flags=re.IGNORECASE,
    )
    if expense_section_match:
        expense_section = prompt[expense_section_match.end() :]
        for match in re.finditer(
            r"([A-Za-zÆØÅæøåÀ-ÿ'’\-\s]+?)\s+(\d[\d\s.,]*)\s*(?:NOK|kr)\b",
            expense_section,
            flags=re.IGNORECASE,
        ):
            description = match.group(1).strip(" ,;:.")
            description = re.sub(
                r"^(?:and|og|et|e)\s+",
                "",
                description,
                flags=re.IGNORECASE,
            ).strip(" ,;:.")
            amount = parse_numeric_string(match.group(2))
            if not description or amount is None:
                continue
            expense_entries.append(
                {
                    "description": description,
                    "amount_currency": amount,
                }
            )

    return CreateTravelExpenseIntent(
        task_type="create_travel_expense",
        title=title,
        employee_first_name=first_name,
        employee_last_name=last_name,
        employee_email=email_match.group(1),
        per_diem_entries=per_diem_entries,
        expense_entries=expense_entries,
    )


def build_hours_invoice_fallback(prompt: str) -> CreateInvoiceIntent | None:
    lowered = normalize_text_for_match(prompt)
    if not any(token in lowered for token in ("log ", "logged hours", "timer", "horas", "heures")):
        return None
    if not any(token in lowered for token in ("invoice", "factura", "faktura", "rechnung", "facture")):
        return None

    customer = extract_invoice_customer(prompt)
    if customer is None:
        return None

    hours_match = re.search(
        r"(\d[\d\s.,]*)\s*(?:hours?|timer|horas|heures)\b",
        prompt,
        flags=re.IGNORECASE,
    )
    rate_match = re.search(
        r"(\d[\d\s.,]*)\s*(?:NOK|kr)\s*/\s*h\b|hourly rate[: ]+(\d[\d\s.,]*)\s*(?:NOK|kr)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not hours_match or not rate_match:
        return None

    hours = parse_numeric_string(hours_match.group(1))
    rate = parse_numeric_string(rate_match.group(1) or rate_match.group(2))
    if hours is None or rate is None:
        return None

    activity_match = re.search(r'activity\s+"([^"]+)"|aktivitet\s+"([^"]+)"', prompt, flags=re.IGNORECASE)
    project_match = re.search(r'project\s+"([^"]+)"|prosjekt\s+"([^"]+)"', prompt, flags=re.IGNORECASE)
    activity_name = next(
        (
            group
            for group in (
                activity_match.group(1) if activity_match else None,
                activity_match.group(2) if activity_match else None,
            )
            if group
        ),
        None,
    )
    project_name = next(
        (
            group
            for group in (
                project_match.group(1) if project_match else None,
                project_match.group(2) if project_match else None,
            )
            if group
        ),
        None,
    )

    description_parts = [activity_name or "Consulting hours"]
    if project_name:
        description_parts.append(project_name)

    return CreateInvoiceIntent(
        task_type="create_invoice",
        customer=customer,
        send_to_customer=False,
        lines=[
            {
                "description": " - ".join(description_parts),
                "quantity": hours,
                "unit_price_excluding_vat_currency": rate,
            }
        ],
    )


def build_fresh_invoice_state_fallback(prompt: str) -> CreateInvoiceIntent | None:
    lowered = normalize_text_for_match(prompt)
    if not any(
        token in lowered
        for token in (
            "invoice",
            "factura",
            "faktura",
            "rechnung",
            "facture",
        )
    ):
        return None

    register_full_payment = any(
        token in lowered
        for token in (
            "register full payment",
            "record full payment",
            "registre le paiement integral",
            "enregistrez le paiement integral",
            "registre el pago completo",
            "registrer full betaling",
            "registrer full betaling",
            "registe o pagamento integral",
            "registe o pagamento total",
            "erfassen sie die zahlung",
            "vollstandige zahlung",
        )
    )
    reverse_payment = any(
        token in lowered
        for token in (
            "reverse the payment",
            "reverser betalingen",
            "reverser betalingen",
            "returned by the bank",
            "returnert av banken",
            "retourne par la banque",
            "zuruckgebucht",
            "devolvida pelo banco",
        )
    )
    if not register_full_payment and not reverse_payment:
        return None

    customer = extract_invoice_customer(prompt)
    if customer is None:
        return None

    description = extract_invoice_line_description(prompt)
    amount = extract_invoice_line_amount(prompt)
    if not description or amount is None:
        return None

    return CreateInvoiceIntent(
        task_type="create_invoice",
        customer=customer,
        register_full_payment=register_full_payment and not reverse_payment,
        send_to_customer=False,
        lines=[
            {
                "description": description,
                "quantity": 1,
                "unit_price_excluding_vat_currency": amount,
            }
        ],
    )


def build_credit_note_fallback(prompt: str) -> CreateInvoiceIntent | None:
    lowered = normalize_text_for_match(prompt)
    if not re.search(
        r"(credit note|credit memo|kreditnota|kreditnote|nota de cr.?dito|note de cr.?dit)",
        lowered,
    ):
        return None

    customer = extract_invoice_customer(prompt)
    if customer is None:
        return None

    description = extract_invoice_line_description(prompt)
    amount = extract_invoice_line_amount(prompt)
    if not description or amount is None:
        return None

    return CreateInvoiceIntent(
        task_type="create_invoice",
        customer=customer,
        is_credit_note=True,
        send_to_customer=False,
        lines=[
            {
                "description": description,
                "quantity": 1,
                "unit_price_excluding_vat_currency": amount,
            }
        ],
    )


def extract_invoice_customer(prompt: str) -> InvoiceCustomerIntent | None:
    org_match = re.search(
        r"(?P<name>[^()]{2,120}?)\s*\((?:[^)]*?)(?:org(?:\.|anization)?[^0-9)]{0,16})(?P<org>\d{9})\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if not org_match:
        return None

    raw_name = org_match.group("name").strip(" ,:")
    raw_name = re.sub(
        r"^(?:the\s+payment\s+from|payment\s+from|betalingen\s+fra|die\s+zahlung\s+von|le\s+paiement\s+de|o\s+pagamento\s+de)\s+",
        "",
        raw_name,
        flags=re.IGNORECASE,
    ).strip(" ,:")
    raw_name = re.sub(
        r"^(?:the\s+customer|customer|kunden?|cliente|client|le\s+client|la\s+cliente)\s+",
        "",
        raw_name,
        flags=re.IGNORECASE,
    ).strip(" ,:")
    raw_name = re.sub(
        r"^.*\b(?:for|til|para|pour|customer|kunden?|cliente|client)\s+",
        "",
        raw_name,
        flags=re.IGNORECASE,
    ).strip(" ,:")
    if not raw_name:
        return None

    return InvoiceCustomerIntent(
        name=raw_name,
        organization_number=org_match.group("org").strip(),
    )


def extract_invoice_line_description(prompt: str) -> str | None:
    match = re.search(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]", prompt)
    if not match:
        return None
    description = match.group(1).strip()
    return description or None


def extract_invoice_line_amount(prompt: str) -> float | None:
    patterns = [
        r"(\d[\d\s.,]*)\s*(?:NOK|kr)\s*(?:excluding|excl\.?|exklusiv|exclusive|ohne|hors|sin|avgiftsfri|mva|mwst|iva|tva)",
        r"(\d[\d\s.,]*)\s*(?:NOK|kr)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        amount = parse_numeric_string(match.group(1))
        if amount is not None:
            return amount
    return None


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
