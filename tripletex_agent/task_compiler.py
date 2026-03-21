from __future__ import annotations

from datetime import date

from tripletex_agent.models import ExecutionPlan, TaskAction
from tripletex_agent.planning_base import PlanningError
from tripletex_agent.task_intents import (
    AddressInput,
    CreateCustomerIntent,
    CreateDepartmentIntent,
    CreateEmployeeIntent,
    CreateInvoiceIntent,
    CreateProjectIntent,
    CreateProductIntent,
    CreateTravelExpenseIntent,
    CreateVoucherIntent,
    DeleteTravelExpenseIntent,
    DeleteDepartmentIntent,
    InvoiceCustomerIntent,
    InvoiceLineIntent,
    SupportedTaskIntent,
    UpdateCustomerIntent,
    UpdateEmployeeIntent,
    UnsupportedTaskIntent,
    VoucherPostingIntent,
)


def compile_task_intent(
    intent: SupportedTaskIntent,
    *,
    allow_beta_endpoints: bool = False,
) -> ExecutionPlan:
    if isinstance(intent, UnsupportedTaskIntent):
        raise PlanningError(intent.reason)
    if isinstance(intent, CreateEmployeeIntent):
        return compile_create_employee(intent, allow_beta_endpoints=allow_beta_endpoints)
    if isinstance(intent, CreateCustomerIntent):
        return compile_create_customer(intent)
    if isinstance(intent, UpdateCustomerIntent):
        return compile_update_customer(intent)
    if isinstance(intent, CreateProductIntent):
        return compile_create_product(intent)
    if isinstance(intent, UpdateEmployeeIntent):
        return compile_update_employee(intent)
    if isinstance(intent, CreateDepartmentIntent):
        return compile_create_department(intent)
    if isinstance(intent, DeleteDepartmentIntent):
        return compile_delete_department(intent)
    if isinstance(intent, CreateVoucherIntent):
        return compile_create_voucher(intent)
    if isinstance(intent, CreateTravelExpenseIntent):
        return compile_create_travel_expense(intent)
    if isinstance(intent, DeleteTravelExpenseIntent):
        return compile_delete_travel_expense(intent)
    if isinstance(intent, CreateProjectIntent):
        return compile_create_project(intent)
    if isinstance(intent, CreateInvoiceIntent):
        return compile_create_invoice(intent)
    raise PlanningError(f"Unsupported intent type: {type(intent).__name__}")


def compile_create_employee(
    intent: CreateEmployeeIntent,
    *,
    allow_beta_endpoints: bool,
) -> ExecutionPlan:
    if (intent.user_type in {"STANDARD", "EXTENDED"} or intent.entitlement_template) and not intent.email:
        raise PlanningError(
            "Employee tasks that require Tripletex access or entitlements must include an email address"
        )

    body: dict[str, object] = {
        "firstName": intent.first_name,
        "lastName": intent.last_name,
    }
    if intent.date_of_birth:
        body["dateOfBirth"] = intent.date_of_birth
    if intent.national_identity_number:
        body["nationalIdentityNumber"] = intent.national_identity_number
    if intent.email:
        body["email"] = intent.email
    if intent.bank_account_number:
        body["bankAccountNumber"] = intent.bank_account_number
    if intent.phone_number_mobile:
        body["phoneNumberMobile"] = intent.phone_number_mobile
    if intent.phone_number_work:
        body["phoneNumberWork"] = intent.phone_number_work
    if intent.employee_number:
        body["employeeNumber"] = intent.employee_number
    comments = [intent.comments] if intent.comments else []
    if intent.position_code:
        comments.append(f"position_code={intent.position_code}")
    if intent.job_title:
        comments.append(f"job_title={intent.job_title}")
    if intent.annual_salary is not None:
        comments.append(f"annual_salary={intent.annual_salary}")
    if intent.employment_percentage is not None:
        comments.append(f"employment_percentage={intent.employment_percentage}")
    if intent.daily_working_hours is not None:
        comments.append(f"daily_working_hours={intent.daily_working_hours}")
    if intent.start_date:
        comments.append(f"start_date={intent.start_date}")
    if comments:
        body["comments"] = " | ".join(comments)
    if intent.address:
        body["address"] = compile_address(intent.address)

    user_type = intent.user_type or "NO_ACCESS"
    requested_beta_entitlements = (
        intent.entitlement_template
        if intent.entitlement_template and intent.entitlement_template != "NONE_PRIVILEGES"
        else None
    )
    if requested_beta_entitlements:
        user_type = "EXTENDED"
    if user_type:
        body["userType"] = user_type

    if intent.department_name:
        actions = [
            TaskAction(
                id="create_employee_department",
                description="Create the requested department for the employee",
                method="POST",
                path="/department",
                body={"name": intent.department_name},
                save_as="department",
            ),
            TaskAction(
                id="create_employee",
                description="Create the employee",
                method="POST",
                path="/employee",
                body={
                    **body,
                    "department": {"id": "{{create_employee_department.value.id}}"},
                },
                save_as="employee",
            ),
        ]
    else:
        actions = [
            TaskAction(
                id="get_departments",
                description="Fetch an active department to satisfy the employee create requirement",
                method="GET",
                path="/department",
                params={
                    "count": 1,
                    "isInactive": False,
                    "fields": "id,name",
                },
            ),
            TaskAction(
                id="create_employee",
                description="Create the employee",
                method="POST",
                path="/employee",
                body={
                    **body,
                    "department": {"id": "{{get_departments.values.0.id}}"},
                },
                save_as="employee",
            ),
        ]
    verification_notes = ["Check created employee fields"]
    if requested_beta_entitlements and allow_beta_endpoints:
        actions.append(
            TaskAction(
                id="grant_employee_entitlements",
                description="Apply the requested Tripletex entitlement template",
                method="PUT",
                path="/employee/entitlement/:grantEntitlementsByTemplate",
                params={
                    "employeeId": "{{create_employee.value.id}}",
                    "template": requested_beta_entitlements,
                },
            )
        )
        verification_notes.append("Check entitlement template if requested")
    elif requested_beta_entitlements:
        verification_notes.append(
            "Requested entitlement template was not applied because beta Tripletex endpoints are disabled"
        )

    return ExecutionPlan(
        goal=f"Create employee {intent.first_name} {intent.last_name}",
        actions=actions,
        verification_notes=verification_notes,
    )


def compile_create_customer(intent: CreateCustomerIntent) -> ExecutionPlan:
    body = compile_customer_body(intent)
    return ExecutionPlan(
        goal=f"Create customer {intent.name}",
        actions=[
            TaskAction(
                id="create_customer",
                description="Create the customer",
                method="POST",
                path="/customer",
                body=body,
                save_as="customer",
            )
        ],
        verification_notes=["Check created customer fields"],
    )


def compile_update_customer(intent: UpdateCustomerIntent) -> ExecutionPlan:
    if not has_any_value(intent.match_name, intent.match_email, intent.match_organization_number):
        raise PlanningError("Customer update tasks require at least one customer matcher")

    update_body = prune_none(
        {
            "email": intent.new_email,
            "invoiceEmail": intent.new_invoice_email,
            "overdueNoticeEmail": intent.new_overdue_notice_email,
            "phoneNumber": intent.new_phone_number,
            "phoneNumberMobile": intent.new_phone_number_mobile,
            "description": intent.new_description,
            "website": intent.new_website,
            "language": intent.new_language,
            "invoicesDueIn": intent.new_invoices_due_in,
            "invoicesDueInType": intent.new_invoices_due_in_type,
            "postalAddress": compile_address(intent.new_postal_address) if intent.new_postal_address else None,
            "physicalAddress": compile_address(intent.new_physical_address)
            if intent.new_physical_address
            else None,
        }
    )
    if not update_body:
        raise PlanningError("Customer update tasks require at least one field to update")

    return ExecutionPlan(
        goal=f"Update customer {intent.match_name or intent.match_email or intent.match_organization_number}",
        actions=[
            TaskAction(
                id="find_customer",
                description="Find the target customer",
                method="GET",
                path="/customer",
                params=prune_none(
                    {
                        "customerName": intent.match_name,
                        "email": intent.match_email,
                        "organizationNumber": intent.match_organization_number,
                        "isInactive": False,
                        "count": 1,
                        "fields": "id,name,email,organizationNumber",
                    }
                ),
            ),
            TaskAction(
                id="update_customer",
                description="Update the matched customer",
                method="PUT",
                path="/customer/{{find_customer.values.0.id}}",
                body=update_body,
                save_as="customer",
            ),
        ],
        verification_notes=["Check the matched customer and updated fields"],
    )


def compile_create_product(intent: CreateProductIntent) -> ExecutionPlan:
    body: dict[str, object] = {"name": intent.name}
    if intent.number:
        body["number"] = intent.number
    if intent.description:
        body["description"] = intent.description
    if intent.order_line_description:
        body["orderLineDescription"] = intent.order_line_description
    if intent.price_excluding_vat_currency is not None:
        body["priceExcludingVatCurrency"] = intent.price_excluding_vat_currency
    if intent.price_including_vat_currency is not None:
        body["priceIncludingVatCurrency"] = intent.price_including_vat_currency
    if intent.cost_excluding_vat_currency is not None:
        body["costExcludingVatCurrency"] = intent.cost_excluding_vat_currency
    if intent.vat_type_id is not None:
        body["vatType"] = {"id": intent.vat_type_id}
    if intent.product_unit_id is not None:
        body["productUnit"] = {"id": intent.product_unit_id}
    if intent.is_stock_item is not None:
        body["isStockItem"] = intent.is_stock_item

    return ExecutionPlan(
        goal=f"Create product {intent.name}",
        actions=[
            TaskAction(
                id="create_product",
                description="Create the product",
                method="POST",
                path="/product",
                body=body,
                save_as="product",
            )
        ],
        verification_notes=["Check created product fields"],
    )


def compile_update_employee(intent: UpdateEmployeeIntent) -> ExecutionPlan:
    if not has_any_value(
        intent.match_first_name,
        intent.match_last_name,
        intent.match_email,
        intent.match_employee_number,
    ):
        raise PlanningError("Employee update tasks require at least one employee matcher")

    update_body = prune_none(
        {
            "email": intent.new_email,
            "phoneNumberMobile": intent.new_phone_number_mobile,
            "phoneNumberWork": intent.new_phone_number_work,
            "comments": intent.new_comments,
            "address": compile_address(intent.new_address) if intent.new_address else None,
            "userType": intent.new_user_type,
        }
    )
    if not update_body:
        raise PlanningError("Employee update tasks require at least one field to update")

    return ExecutionPlan(
        goal=(
            "Update employee "
            f"{intent.match_first_name or ''} {intent.match_last_name or ''}".strip()
            or intent.match_email
            or intent.match_employee_number
            or "employee"
        ),
        actions=[
            TaskAction(
                id="find_employee",
                description="Find the target employee",
                method="GET",
                path="/employee",
                params=prune_none(
                    {
                        "firstName": intent.match_first_name,
                        "lastName": intent.match_last_name,
                        "email": intent.match_email,
                        "employeeNumber": intent.match_employee_number,
                        "count": 1,
                        "fields": "id,firstName,lastName,email,employeeNumber",
                    }
                ),
            ),
            TaskAction(
                id="get_employee_details",
                description="Fetch current employee details needed for a safe update",
                method="GET",
                path="/employee/{{find_employee.values.0.id}}",
                params={
                    "fields": (
                        "id,firstName,lastName,dateOfBirth,email,phoneNumberMobile,"
                        "phoneNumberWork,comments,address(*),department(id),userType,employeeNumber"
                    )
                },
            ),
            TaskAction(
                id="update_employee",
                description="Update the matched employee",
                method="PUT",
                path="/employee/{{find_employee.values.0.id}}",
                body={
                    "firstName": "{{get_employee_details.value.firstName}}",
                    "lastName": "{{get_employee_details.value.lastName}}",
                    "dateOfBirth": intent.new_date_of_birth
                    or "{{get_employee_details.value.dateOfBirth}}",
                    "email": intent.new_email or "{{get_employee_details.value.email}}",
                    "phoneNumberMobile": intent.new_phone_number_mobile
                    or "{{get_employee_details.value.phoneNumberMobile}}",
                    "phoneNumberWork": intent.new_phone_number_work
                    or "{{get_employee_details.value.phoneNumberWork}}",
                    "comments": intent.new_comments or "{{get_employee_details.value.comments}}",
                    "address": intent.new_address and compile_address(intent.new_address)
                    or "{{get_employee_details.value.address}}",
                    "department": {"id": "{{get_employee_details.value.department.id}}"},
                    "userType": intent.new_user_type or "{{get_employee_details.value.userType}}",
                    "employeeNumber": "{{get_employee_details.value.employeeNumber}}",
                },
                save_as="employee",
            ),
        ],
        verification_notes=["Check the matched employee and updated fields"],
    )


def compile_create_department(intent: CreateDepartmentIntent) -> ExecutionPlan:
    body = prune_none(
        {
            "name": intent.name,
            "departmentNumber": intent.department_number,
            "businessActivityTypeId": intent.business_activity_type_id,
        }
    )
    return ExecutionPlan(
        goal=f"Create department {intent.name}",
        actions=[
            TaskAction(
                id="create_department",
                description="Create the department",
                method="POST",
                path="/department",
                body=body,
                save_as="department",
            )
        ],
        verification_notes=["Check created department fields"],
    )


def compile_delete_department(intent: DeleteDepartmentIntent) -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Delete department {intent.match_name}",
        actions=[
            TaskAction(
                id="find_department",
                description="Find the target department",
                method="GET",
                path="/department",
                params={
                    "name": intent.match_name,
                    "isInactive": False,
                    "count": 1,
                    "fields": "id,name",
                },
            ),
            TaskAction(
                id="delete_department",
                description="Delete the matched department",
                method="DELETE",
                path="/department/{{find_department.values.0.id}}",
            ),
        ],
        verification_notes=["Check that the department no longer exists"],
    )


def compile_create_voucher(intent: CreateVoucherIntent) -> ExecutionPlan:
    if len(intent.postings) < 2:
        raise PlanningError("Voucher tasks require at least two postings")

    debit_total = sum(
        posting.amount for posting in intent.postings if posting.entry_type == "DEBIT"
    )
    credit_total = sum(
        posting.amount for posting in intent.postings if posting.entry_type == "CREDIT"
    )
    if round(debit_total - credit_total, 2) != 0:
        raise PlanningError("Voucher postings must balance between debit and credit")

    voucher_date = intent.voucher_date or iso_today()
    unique_account_numbers = []
    for posting in intent.postings:
        if posting.account_number not in unique_account_numbers:
            unique_account_numbers.append(posting.account_number)

    actions: list[TaskAction] = [
        TaskAction(
            id="list_ledger_accounts",
            description="Fetch chart of accounts so voucher postings can reference account IDs",
            method="GET",
            path="/ledger/account",
            params={
                "count": 10000,
                "fields": "id,number,name,ledgerType,vatType(id),legalVatTypes(id),vatLocked",
            },
        )
    ]

    customer_reference: dict[str, object] | None = None
    if intent.customer_name or intent.customer_organization_number:
        actions.append(
            TaskAction(
                id="create_voucher_customer",
                description="Create the customer referenced by the voucher",
                method="POST",
                path="/customer",
                body=prune_none(
                    {
                        "name": intent.customer_name or f"Customer {intent.customer_organization_number}",
                        "organizationNumber": intent.customer_organization_number,
                        "isCustomer": True,
                        "isSupplier": False,
                    }
                ),
                save_as="customer",
            )
        )
        customer_reference = {"id": "{{create_voucher_customer.value.id}}"}

    supplier_reference: dict[str, object] | None = None
    if intent.supplier_invoice_details:
        actions.append(
            TaskAction(
                id="create_voucher_supplier",
                description="Create the supplier referenced by the supplier invoice",
                method="POST",
                path="/customer",
                body=prune_none(
                    {
                        "name": intent.supplier_invoice_details.supplier_name,
                        "organizationNumber": intent.supplier_invoice_details.organization_number,
                        "isCustomer": False,
                        "isSupplier": True,
                    }
                ),
                save_as="supplier",
            )
        )
        supplier_reference = {"id": "{{create_voucher_supplier.value.id}}"}

    department_action_ids: dict[str, str] = {}
    for index, department_name in enumerate(
        dict.fromkeys(
            posting.department_name for posting in intent.postings if posting.department_name
        ),
        start=1,
    ):
        if not department_name:
            continue
        action_id = f"create_voucher_department_{index}"
        department_action_ids[department_name] = action_id
        actions.append(
            TaskAction(
                id=action_id,
                description=f"Create department {department_name} for voucher dimensioning",
                method="POST",
                path="/department",
                body={"name": department_name},
                save_as=f"voucher_department_{index}",
            )
        )

    for account_number in unique_account_numbers:
        actions.append(
            TaskAction(
                id=f"select_account_{account_number}",
                description=f"Select ledger account {account_number}",
                method="SELECT",
                path="select",
                body={
                    "source": "{{list_ledger_accounts.values}}",
                    "criteria": {"number": account_number},
                },
            )
        )

    actions.append(
        TaskAction(
            id="create_voucher",
            description="Create the voucher with the requested postings",
            method="POST",
            path="/ledger/voucher",
            body=prune_none(
                {
                    "date": voucher_date,
                    "description": intent.description or "Voucher created by automation",
                    "postings": [
                        compile_voucher_posting(
                            posting,
                            row=index,
                            voucher_date=voucher_date,
                            customer_reference=customer_reference,
                            supplier_reference=supplier_reference,
                            department_action_ids=department_action_ids,
                        )
                        for index, posting in enumerate(intent.postings, start=1)
                    ],
                }
            ),
            save_as="voucher",
        )
    )

    return ExecutionPlan(
        goal=intent.description or f"Create voucher dated {voucher_date}",
        actions=actions,
        verification_notes=["Check that the voucher was created with balanced postings"],
    )


def compile_create_travel_expense(intent: CreateTravelExpenseIntent) -> ExecutionPlan:
    travel_details = compile_travel_expense_details(intent)
    actions = [
        TaskAction(
            id="get_departments_for_travel_expense",
            description="Fetch an active department to satisfy the employee create requirement",
            method="GET",
            path="/department",
            params={
                "count": 1,
                "isInactive": False,
                "fields": "id,name",
            },
        ),
        TaskAction(
            id="create_employee_for_travel_expense",
            description="Create the employee for the travel expense",
            method="POST",
            path="/employee",
            body=compile_employee_body_for_travel_expense(intent),
            save_as="employee",
        ),
        TaskAction(
            id="create_travel_expense",
            description="Create the travel expense",
            method="POST",
            path="/travelExpense",
            body=prune_none(
                {
                    "title": intent.title,
                    "employee": {"id": "{{create_employee_for_travel_expense.value.id}}"},
                    "travelDetails": travel_details,
                }
            ),
            save_as="travel_expense",
        ),
    ]
    return ExecutionPlan(
        goal=f"Create travel expense {intent.title}",
        actions=actions,
        verification_notes=["Check created employee fields", "Check created travel expense fields"],
    )


def compile_delete_travel_expense(intent: DeleteTravelExpenseIntent) -> ExecutionPlan:
    criteria = {
        "title": intent.title,
        "employee.firstName": intent.employee_first_name,
        "employee.lastName": intent.employee_last_name,
    }
    if intent.employee_email:
        criteria["employee.email"] = intent.employee_email

    return ExecutionPlan(
        goal=f"Delete travel expense {intent.title}",
        actions=[
            TaskAction(
                id="find_travel_expenses",
                description="List travel expenses to find the requested item",
                method="GET",
                path="/travelExpense",
                params={
                    "state": "ALL",
                    "count": 100,
                    "fields": (
                        "id,title,date,state,employee(id,firstName,lastName,email),"
                        "travelDetails(departureDate,returnDate)"
                    ),
                },
            ),
            TaskAction(
                id="select_travel_expense",
                description="Select the travel expense with the requested title",
                method="SELECT",
                path="select",
                body={
                    "source": "{{find_travel_expenses.values}}",
                    "criteria": criteria,
                },
            ),
            TaskAction(
                id="delete_travel_expense",
                description="Delete the matched travel expense",
                method="DELETE",
                path="/travelExpense/{{select_travel_expense.id}}",
            ),
        ],
        verification_notes=["Check that the travel expense no longer exists"],
    )


def compile_create_project(intent: CreateProjectIntent) -> ExecutionPlan:
    start_date = intent.start_date or iso_today()
    actions: list[TaskAction] = []
    customer_reference: dict[str, object] | None = None
    if intent.customer:
        actions.append(
            TaskAction(
                id="create_project_customer",
                description="Create the project customer",
                method="POST",
                path="/customer",
                body=compile_customer_body(intent.customer),
                save_as="customer",
            )
        )
        customer_reference = {"id": "{{create_project_customer.value.id}}"}

    project_manager_reference = {"id": "{{find_project_manager.values.0.id}}"}
    if any(
        value
        for value in (
            intent.project_manager_email,
            intent.project_manager_first_name,
            intent.project_manager_last_name,
        )
    ):
        actions.extend(
            [
                TaskAction(
                    id="get_departments_for_project_manager",
                    description="Fetch an active department to satisfy employee create requirements",
                    method="GET",
                    path="/department",
                    params={
                        "count": 1,
                        "isInactive": False,
                        "fields": "id,name",
                    },
                ),
                TaskAction(
                    id="create_project_manager",
                    description="Create the project manager employee when the prompt provides a named manager",
                    method="POST",
                    path="/employee",
                    body=prune_none(
                        {
                            "firstName": intent.project_manager_first_name or "Project",
                            "lastName": intent.project_manager_last_name or "Manager",
                            "email": intent.project_manager_email,
                            "userType": "NO_ACCESS",
                            "department": {"id": "{{get_departments_for_project_manager.values.0.id}}"},
                        }
                    ),
                    save_as="project_manager",
                ),
            ]
        )
        project_manager_reference = {"id": "{{create_project_manager.value.id}}"}
    else:
        actions.append(
            TaskAction(
                id="find_project_manager",
                description="Find an employee to assign as project manager",
                method="GET",
                path="/employee",
                params={
                    "count": 1,
                    "fields": "id,firstName,lastName,email",
                },
            )
        )

    actions.append(
        TaskAction(
            id="create_project",
            description="Create the project",
            method="POST",
            path="/project",
            body=prune_none(
                {
                    "name": intent.name,
                    "number": intent.number,
                    "description": intent.description,
                    "startDate": start_date,
                    "endDate": intent.end_date,
                    "reference": intent.reference,
                    "invoiceComment": intent.invoice_comment,
                    "isInternal": intent.is_internal,
                    "isOffer": intent.is_offer,
                    "isFixedPrice": intent.is_fixed_price or intent.fixed_price_amount is not None,
                    "customer": customer_reference,
                    "projectManager": project_manager_reference,
                }
            ),
            save_as="project",
        )
    )

    notes = ["Check created project fields"]
    if intent.customer:
        notes.insert(0, "Check created customer fields")

    return ExecutionPlan(
        goal=f"Create project {intent.name}",
        actions=actions,
        verification_notes=notes,
    )


def compile_create_invoice(intent: CreateInvoiceIntent) -> ExecutionPlan:
    if not intent.lines:
        raise PlanningError("Invoice tasks require at least one invoice line")

    invoice_date = intent.invoice_date or iso_today()
    invoice_due_date = intent.invoice_due_date or invoice_date
    order_date = intent.order_date or invoice_date
    delivery_date = intent.delivery_date or invoice_date
    prioritize_amounts_including_vat = all(
        line.unit_price_including_vat_currency is not None
        and line.unit_price_excluding_vat_currency is None
        for line in intent.lines
    )

    actions = [
        TaskAction(
            id="create_customer",
            description="Create the invoice customer",
            method="POST",
            path="/customer",
            body=compile_customer_body(intent.customer),
            save_as="customer",
        ),
        TaskAction(
            id="create_order",
            description="Create an order for the invoice lines",
            method="POST",
            path="/order",
            body=prune_none(
                {
                    "customer": {"id": "{{create_customer.value.id}}"},
                    "orderDate": order_date,
                    "deliveryDate": delivery_date,
                    "isPrioritizeAmountsIncludingVat": prioritize_amounts_including_vat,
                    "reference": intent.order_reference,
                    "orderLines": [compile_invoice_line(line) for line in intent.lines],
                }
            ),
            save_as="order",
        ),
        TaskAction(
            id="create_invoice",
            description="Create the invoice from the order",
            method="POST",
            path="/invoice",
            params={"sendToCustomer": intent.send_to_customer},
            body=prune_none(
                {
                    "invoiceDate": invoice_date,
                    "invoiceDueDate": invoice_due_date,
                    "customer": {"id": "{{create_customer.value.id}}"},
                    "invoiceComment": intent.invoice_comment,
                    "deliveryDate": delivery_date,
                    "orders": [{"id": "{{create_order.value.id}}"}],
                }
            ),
            save_as="invoice",
        ),
    ]

    return ExecutionPlan(
        goal=f"Create invoice for customer {intent.customer.name}",
        actions=actions,
        verification_notes=["Check customer, order, and invoice creation"],
    )


def compile_customer_body(intent: CreateCustomerIntent | InvoiceCustomerIntent) -> dict[str, object]:
    body: dict[str, object] = {
        "name": intent.name,
        "isCustomer": intent.is_customer,
        "isSupplier": intent.is_supplier,
    }
    if intent.email:
        body["email"] = intent.email
    if intent.invoice_email:
        body["invoiceEmail"] = intent.invoice_email
    if intent.overdue_notice_email:
        body["overdueNoticeEmail"] = intent.overdue_notice_email
    if intent.phone_number:
        body["phoneNumber"] = intent.phone_number
    if intent.phone_number_mobile:
        body["phoneNumberMobile"] = intent.phone_number_mobile
    if intent.organization_number:
        body["organizationNumber"] = intent.organization_number
    if intent.description:
        body["description"] = intent.description
    if intent.website:
        body["website"] = intent.website
    if intent.language:
        body["language"] = intent.language
    if intent.invoices_due_in is not None:
        body["invoicesDueIn"] = intent.invoices_due_in
    if intent.invoices_due_in_type:
        body["invoicesDueInType"] = intent.invoices_due_in_type
    if intent.postal_address:
        body["postalAddress"] = compile_address(intent.postal_address)
    if intent.physical_address:
        body["physicalAddress"] = compile_address(intent.physical_address)
    return body


def compile_employee_body_for_travel_expense(intent: CreateTravelExpenseIntent) -> dict[str, object]:
    body: dict[str, object] = {
        "firstName": intent.employee_first_name,
        "lastName": intent.employee_last_name,
        "userType": "NO_ACCESS",
    }
    if intent.employee_email:
        body["email"] = intent.employee_email
    return {
        **body,
        "department": {"id": "{{get_departments_for_travel_expense.values.0.id}}"},
    }


def compile_travel_expense_details(details: object) -> dict[str, object] | None:
    from tripletex_agent.task_intents import CreateTravelExpenseIntent, TravelExpenseDetailsIntent

    expense_summary: str | None = None
    if isinstance(details, CreateTravelExpenseIntent):
        intent = details
        details = intent.details
        expense_summary = summarize_travel_expense_entries(intent)

    if not isinstance(details, TravelExpenseDetailsIntent):
        if expense_summary:
            return {"departureDate": iso_today(), "returnDate": iso_today(), "purpose": expense_summary}
        return None

    departure_date = details.departure_date or iso_today()
    purpose = details.purpose
    if expense_summary:
        purpose = f"{purpose} | {expense_summary}" if purpose else expense_summary
    return prune_none(
        {
            "departureDate": departure_date,
            "returnDate": details.return_date or departure_date,
            "departureFrom": details.departure_from,
            "destination": details.destination,
            "purpose": purpose,
            "isDayTrip": details.is_day_trip,
            "isForeignTravel": details.is_foreign_travel,
            "isCompensationFromRates": details.is_compensation_from_rates,
        }
    )


def compile_invoice_line(line: InvoiceLineIntent) -> dict[str, object]:
    body: dict[str, object] = {
        "description": line.description,
        "count": line.quantity,
    }
    if line.unit_price_excluding_vat_currency is not None:
        body["unitPriceExcludingVatCurrency"] = line.unit_price_excluding_vat_currency
    if line.unit_price_including_vat_currency is not None:
        body["unitPriceIncludingVatCurrency"] = line.unit_price_including_vat_currency
    if line.discount_percent is not None:
        body["discount"] = line.discount_percent
    if line.vat_type_id is not None:
        body["vatType"] = {"id": line.vat_type_id}
    return body


def compile_voucher_posting(
    posting: VoucherPostingIntent,
    *,
    row: int,
    voucher_date: str,
    customer_reference: dict[str, object] | None = None,
    supplier_reference: dict[str, object] | None = None,
    department_action_ids: dict[str, str] | None = None,
) -> dict[str, object]:
    signed_amount = posting.amount if posting.entry_type == "DEBIT" else -posting.amount
    return prune_none(
        {
            "row": row,
            "date": voucher_date,
            "amountGross": signed_amount,
            "amountGrossCurrency": signed_amount,
            "description": posting.description,
            "account": {"id": f"{{{{select_account_{posting.account_number}.id}}}}"},
            "customer": customer_reference
            if customer_reference and is_likely_customer_ledger_account(posting.account_number)
            else None,
            "supplier": supplier_reference
            if supplier_reference and is_likely_supplier_ledger_account(posting.account_number)
            else None,
            "department": {"id": f"{{{{{department_action_ids[posting.department_name]}.value.id}}}}"}
            if department_action_ids and posting.department_name in department_action_ids
            else None,
            "vatType": {"id": posting.vat_type_id} if posting.vat_type_id is not None else None,
        }
    )


def compile_address(address: AddressInput) -> dict[str, object]:
    body: dict[str, object] = {"addressLine1": address.address_line1}
    if address.address_line2:
        body["addressLine2"] = address.address_line2
    if address.postal_code:
        body["postalCode"] = address.postal_code
    if address.city:
        body["city"] = address.city
    if address.country_id is not None:
        body["country"] = {"id": address.country_id}
    return body


def is_likely_supplier_ledger_account(account_number: int) -> bool:
    return 2400 <= account_number <= 2499


def is_likely_customer_ledger_account(account_number: int) -> bool:
    return 1500 <= account_number <= 1599


def summarize_travel_expense_entries(intent: CreateTravelExpenseIntent) -> str | None:
    segments: list[str] = []
    for entry in intent.per_diem_entries:
        if not isinstance(entry, dict):
            continue
        days = entry.get("number_of_days")
        amount = entry.get("amount_per_day_currency")
        if days and amount:
            segments.append(f"Per diem: {days} days x {amount} NOK")
    for entry in intent.expense_entries:
        if not isinstance(entry, dict):
            continue
        description = entry.get("description") or "Expense"
        amount = entry.get("amount_currency") or entry.get("amount")
        if amount:
            segments.append(f"{description}: {amount} NOK")
    if not segments:
        return None
    return "; ".join(str(segment) for segment in segments)


def iso_today() -> str:
    return date.today().isoformat()


def has_any_value(*values: object) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True
    return False


def prune_none(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if item is not None}
