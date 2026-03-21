# AI-NM26 Tripletex

Working repository for the Tripletex case in the Norwegian AI Championship 2026.

## What This Task Is

We need to host an HTTPS `POST /solve` endpoint that receives:

- a natural-language accounting task prompt
- optional file attachments
- Tripletex proxy credentials

Our agent must interpret the task, perform the correct Tripletex API calls through the provided proxy, and return:

```json
{"status":"completed"}
```

## Hard Requirements

- Endpoint must be HTTPS
- Timeout is 300 seconds
- Use the provided `tripletex_credentials.base_url`
- Authenticate to Tripletex with Basic Auth:
  - username: `0`
  - password: `session_token`
- Each competition submission starts from a fresh Tripletex sandbox account
- Prompts can arrive in 7 languages
- Some tasks include attachments such as PDFs or images

## Scoring Priorities

The score is driven by:

1. Correctness
2. Task tier multiplier
3. Efficiency bonus for perfect runs

Efficiency depends on:

- few API calls
- few or no 4xx errors
- avoiding unnecessary verification or trial-and-error

Implication: correctness comes first, but the final design must be deliberate and low-noise.

## What The Agent Must Do Well

- Parse multilingual prompts into a structured task representation
- Extract data from attached files when needed
- Resolve existing entities and create prerequisites when needed
- Execute multi-step Tripletex workflows reliably
- Verify important end-state fields without wasting calls
- Recover from validation errors in a targeted way

## Recommended Architecture

Use a hybrid design:

1. FastAPI service layer
2. Request/auth validation and temporary file handling
3. Attachment extraction layer
4. LLM normalization step that converts prompt plus extracted file data into structured intent
5. Deterministic executor over a typed Tripletex client
6. Minimal verification layer
7. Structured logging for replay and debugging

The LLM should decide intent and parameters. Business execution should stay mostly deterministic.

## Hosted Runtime

The intended production shape is:

- Cloud Run for the public HTTPS `POST /solve` endpoint
- Vertex AI Gemini for multilingual prompt planning
- Application Default Credentials for Google API access
- A user-managed Cloud Run service account with only the permissions the service needs

On Cloud Run, we should use service identity instead of setting `GOOGLE_APPLICATION_CREDENTIALS`.

## Competition Constraints We Must Respect

- AI coding assistants and open-source software are allowed
- Do not share competition-specific solutions, observations, or credentials outside the team
- Do not hardcode or pre-compute responses for specific test cases
- Do not try to extract hidden test data, evaluation logic, or other protected platform details
- Do not circumvent submission limits, cooldowns, or quotas
- Repository must be public before the deadline to remain prize-eligible
- Prize-eligible code must be open-sourced under MIT or an equivalent permissive license

## Important Notes About The Provided Docs

- `docs/AI-NM26_Documentation.md` contains material for multiple competition tasks; only the Tripletex sections are relevant here
- `docs/AI-NM26_Rules.md` contains official competition rules first, then repo-specific working notes appended after the official rules
- One Tripletex section says the prompt is Norwegian, but later sections say prompts come in 7 languages; we should treat the task as multilingual from day one

## Immediate Build Plan

1. Scaffold the FastAPI `POST /solve` service
2. Add a typed Tripletex client with request/response logging and error parsing
3. Define a structured task schema for prompt normalization
4. Implement a first executor for Tier 1 flows
5. Add sandbox replay scripts and fixtures
6. Expand coverage task family by task family

## Current Scaffold

The repo now includes:

- a FastAPI service with `POST /solve` and `GET /healthz`
- typed request and plan models
- attachment materialization with basic PDF/text extraction
- a generic Tripletex API client
- a deterministic plan executor with template substitution between actions
- run artifact logging under `.work/runs/`

The multilingual planner is intentionally still a separate layer. Right now the code supports:

- `stub` planner mode: returns a clear planning error
- `json_prompt` planner mode: lets us test the executor by embedding an `ExecutionPlan` JSON object directly in the prompt
- `gemini_vertex` planner mode: uses Vertex AI Gemini through the official `google-genai` SDK

The Gemini path now extracts a supported typed task intent, then deterministic code compiles it into canonical Tripletex actions. This is safer than letting the model invent raw API calls directly.

## Current Task Coverage

Implemented first-pass support:

- create employee
- create customer
- update customer
- create product
- update employee
- create department
- delete department
- create travel expense
- delete travel expense
- create project
- create invoice

Current invoice flow is canonical:

1. create customer
2. create order
3. create invoice from the order

Current employee flow is safe-by-default:

1. create employee
2. only grant entitlement templates through `/employee/entitlement/:grantEntitlementsByTemplate` when beta endpoints are explicitly enabled

Still unsupported in the current code:

- delete/reverse flows
- travel expense attachment workflows
- module setup tasks
- payment registration and credit notes
- ledger and voucher workflows

## Live Validation

Live sandbox validation completed on March 20, 2026:

- employee creation succeeded
- employee admin entitlement grant succeeded in sandbox when beta endpoints were enabled
- customer creation succeeded
- product creation succeeded
- department creation succeeded
- department deletion succeeded
- travel expense creation succeeded with minimal title plus employee flow, and with optional travel details
- travel expense deletion succeeded by matching title plus employee fields through local selection
- customer update succeeded
- project creation succeeded after assigning a project manager and defaulting startDate
- employee update succeeded when the prompt supplied a birth date for a target employee missing dateOfBirth

Observed sandbox-specific blocker on March 20, 2026:

- invoice creation reached `POST /invoice`, but the sandbox rejected it because the company did not have a registered bank account number
- replaying the same invoice flow with `sendToCustomer=false` on March 21, 2026 still produced the same bank-account validation error
- updating a sandbox-created employee without an existing birth date still fails unless the prompt provides a birth date, because Tripletex validates `dateOfBirth` on update

That means the invoice compiler path is structurally working through customer and order creation, but full invoice validation still depends on resolving company-level invoicing prerequisites in the sandbox.

## Local Development

Install dependencies:

```bash
pip install -e .
```

Run locally:

```bash
uvicorn tripletex_agent.main:app --reload
```

Send a local sandbox test request:

```bash
python scripts/run_local_sandbox_test.py \
  "Create a customer named Acme AS with email post@acme.no." \
  --base-url https://kkpqfuj-amager.tripletex.dev/v2 \
  --session-token YOUR_SESSION_TOKEN
```

Useful environment variables:

- `TRIPLETEX_AGENT_API_KEY`: optional Bearer token required by our endpoint
- `TRIPLETEX_AGENT_PLANNER_MODE`: `stub`, `json_prompt`, or `gemini_vertex`
- `TRIPLETEX_AGENT_ALLOW_BETA_ENDPOINTS`: defaults to `false`; set to `true` only for sandbox experiments that intentionally use beta Tripletex endpoints
- `TRIPLETEX_AGENT_RUNS_DIR`: defaults to `.work/runs`
- `TRIPLETEX_AGENT_TIMEOUT_SECONDS`: per-call timeout for Tripletex API requests
- `GOOGLE_CLOUD_PROJECT`: GCP project for Vertex AI
- `GOOGLE_CLOUD_LOCATION`: Vertex AI location, default `global`
- `TRIPLETEX_AGENT_GEMINI_MODEL`: default `gemini-2.5-flash`
- `TRIPLETEX_AGENT_GEMINI_MAX_ATTACHMENT_BYTES`: default `8000000`
- `TRIPLETEX_AGENT_GEMINI_ATTACHMENT_TEXT_CHARS`: default `12000`

## Cloud Run

Container build is supported through `Dockerfile`.

Example deployment shape:

```bash
gcloud run deploy ai-nm26-tripletex \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --set-env-vars TRIPLETEX_AGENT_PLANNER_MODE=gemini_vertex,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=global
```

Recommended follow-up hardening:

- require an endpoint Bearer token with `TRIPLETEX_AGENT_API_KEY`
- deploy with a dedicated user-managed service account
- grant only the Vertex AI permissions needed by the service
