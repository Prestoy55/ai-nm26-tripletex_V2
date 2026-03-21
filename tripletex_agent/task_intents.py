from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class AddressInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    address_line1: str
    address_line2: str | None = None
    postal_code: str | None = None
    city: str | None = None
    country_id: int | None = None


class UnsupportedTaskIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["unsupported"]
    reason: str


class CreateEmployeeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_employee"]
    first_name: str
    last_name: str
    email: str | None = None
    phone_number_mobile: str | None = None
    phone_number_work: str | None = None
    employee_number: str | None = None
    user_type: Literal["STANDARD", "EXTENDED", "NO_ACCESS"] | None = None
    entitlement_template: (
        Literal[
            "NONE_PRIVILEGES",
            "ALL_PRIVILEGES",
            "INVOICING_MANAGER",
            "PERSONELL_MANAGER",
            "ACCOUNTANT",
            "AUDITOR",
            "DEPARTMENT_LEADER",
        ]
        | None
    ) = None
    comments: str | None = None
    address: AddressInput | None = None


class CreateCustomerIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_customer"]
    name: str
    email: str | None = None
    invoice_email: str | None = None
    overdue_notice_email: str | None = None
    phone_number: str | None = None
    phone_number_mobile: str | None = None
    organization_number: str | None = None
    description: str | None = None
    website: str | None = None
    invoices_due_in: int | None = None
    invoices_due_in_type: str | None = None
    language: str | None = None
    postal_address: AddressInput | None = None
    physical_address: AddressInput | None = None
    is_customer: bool = True
    is_supplier: bool = False


class UpdateCustomerIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["update_customer"]
    match_name: str | None = None
    match_email: str | None = None
    match_organization_number: str | None = None
    new_email: str | None = None
    new_invoice_email: str | None = None
    new_overdue_notice_email: str | None = None
    new_phone_number: str | None = None
    new_phone_number_mobile: str | None = None
    new_description: str | None = None
    new_website: str | None = None
    new_language: str | None = None
    new_invoices_due_in: int | None = None
    new_invoices_due_in_type: str | None = None
    new_postal_address: AddressInput | None = None
    new_physical_address: AddressInput | None = None


class InvoiceCustomerIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    email: str | None = None
    invoice_email: str | None = None
    overdue_notice_email: str | None = None
    phone_number: str | None = None
    phone_number_mobile: str | None = None
    organization_number: str | None = None
    description: str | None = None
    website: str | None = None
    invoices_due_in: int | None = None
    invoices_due_in_type: str | None = None
    language: str | None = None
    postal_address: AddressInput | None = None
    physical_address: AddressInput | None = None
    is_customer: bool = True
    is_supplier: bool = False


class CreateProductIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_product"]
    name: str
    number: str | None = None
    description: str | None = None
    order_line_description: str | None = None
    price_excluding_vat_currency: float | None = None
    price_including_vat_currency: float | None = None
    cost_excluding_vat_currency: float | None = None
    vat_type_id: int | None = None
    product_unit_id: int | None = None
    is_stock_item: bool | None = None


class UpdateEmployeeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["update_employee"]
    match_first_name: str | None = None
    match_last_name: str | None = None
    match_email: str | None = None
    match_employee_number: str | None = None
    new_date_of_birth: str | None = None
    new_email: str | None = None
    new_phone_number_mobile: str | None = None
    new_phone_number_work: str | None = None
    new_comments: str | None = None
    new_address: AddressInput | None = None
    new_user_type: Literal["STANDARD", "EXTENDED", "NO_ACCESS"] | None = None


class CreateDepartmentIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_department"]
    name: str
    department_number: str | None = None
    business_activity_type_id: int | None = None


class DeleteDepartmentIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["delete_department"]
    match_name: str


class TravelExpenseDetailsIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    departure_date: str | None = None
    return_date: str | None = None
    departure_from: str | None = None
    destination: str | None = None
    purpose: str | None = None
    is_day_trip: bool | None = None
    is_foreign_travel: bool | None = None
    is_compensation_from_rates: bool | None = None


class CreateTravelExpenseIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_travel_expense"]
    title: str
    employee_first_name: str
    employee_last_name: str
    employee_email: str | None = None
    details: TravelExpenseDetailsIntent | None = None


class DeleteTravelExpenseIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["delete_travel_expense"]
    title: str
    employee_first_name: str
    employee_last_name: str
    employee_email: str | None = None


class CreateProjectIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_project"]
    name: str
    number: str | None = None
    description: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    reference: str | None = None
    invoice_comment: str | None = None
    is_internal: bool | None = None
    is_offer: bool | None = None
    is_fixed_price: bool | None = None
    customer: InvoiceCustomerIntent | None = None


class InvoiceLineIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: float = Field(default=1)
    unit_price_excluding_vat_currency: float | None = None
    unit_price_including_vat_currency: float | None = None
    discount_percent: float | None = None
    vat_type_id: int | None = None


class CreateInvoiceIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["create_invoice"]
    customer: InvoiceCustomerIntent
    invoice_date: str | None = None
    invoice_due_date: str | None = None
    order_date: str | None = None
    delivery_date: str | None = None
    invoice_comment: str | None = None
    order_reference: str | None = None
    send_to_customer: bool = False
    lines: list[InvoiceLineIntent] = Field(default_factory=list)


SupportedTaskIntent = Annotated[
    UnsupportedTaskIntent
    | CreateEmployeeIntent
    | CreateCustomerIntent
    | UpdateCustomerIntent
    | CreateProductIntent
    | UpdateEmployeeIntent
    | CreateDepartmentIntent
    | DeleteDepartmentIntent
    | CreateTravelExpenseIntent
    | DeleteTravelExpenseIntent
    | CreateProjectIntent
    | CreateInvoiceIntent,
    Field(discriminator="task_type"),
]

SUPPORTED_TASK_INTENT_ADAPTER = TypeAdapter(SupportedTaskIntent)
