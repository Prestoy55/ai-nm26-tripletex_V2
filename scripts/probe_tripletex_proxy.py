from __future__ import annotations

import argparse
import json
from typing import Any

from tripletex_agent.tripletex_client import TripletexApiError, TripletexClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe the Tripletex proxy directly using sandbox credentials."
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Tripletex base URL, for example https://kkpqfuj-amager.tripletex.dev/v2",
    )
    parser.add_argument(
        "--session-token",
        required=True,
        help="Tripletex session token",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    accounts = subparsers.add_parser(
        "accounts",
        help="List or search ledger accounts",
    )
    accounts.add_argument("--number", type=int, help="Exact account number to search for")
    accounts.add_argument(
        "--prefix",
        help="Account-number prefix to search for, for example 12",
    )
    accounts.add_argument(
        "--name-contains",
        help="Case-insensitive substring to search for in account names",
    )
    accounts.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of matching accounts to print",
    )

    product = subparsers.add_parser(
        "product-create",
        help="Try a minimal product create to test proxy permissions",
    )
    product.add_argument("--name", default="Codex Probe Product")
    product.add_argument("--number", default="CODEX-PROBE-1")
    product.add_argument("--price", type=float, default=100.0)
    product.add_argument("--vat-type-id", type=int, default=3)

    request = subparsers.add_parser(
        "request",
        help="Send a raw request to the proxy",
    )
    request.add_argument("method", choices=["GET", "POST", "PUT", "DELETE"])
    request.add_argument("path", help="Path like /customer or /invoice/paymentType")
    request.add_argument("--params", help="JSON object of query params")
    request.add_argument("--body", help="JSON object body")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with TripletexClient(
        base_url=args.base_url,
        session_token=args.session_token,
        timeout_seconds=30,
    ) as client:
        try:
            if args.command == "accounts":
                return run_accounts_probe(client, args)
            if args.command == "product-create":
                return run_product_probe(client, args)
            if args.command == "request":
                return run_raw_request(client, args)
        except TripletexApiError as exc:
            print(exc)
            if isinstance(exc.response_body, (dict, list)):
                print(json.dumps(exc.response_body, ensure_ascii=False, indent=2))
            else:
                print(exc.response_body)
            return 1

    return 0


def run_accounts_probe(client: TripletexClient, args: argparse.Namespace) -> int:
    _, payload = client.request(
        "GET",
        "/ledger/account",
        params={
            "count": 10000,
            "fields": "id,number,name,ledgerType,vatType(id),legalVatTypes(id),vatLocked",
        },
    )
    values = payload.get("values", []) if isinstance(payload, dict) else []
    matches: list[dict[str, Any]] = []

    for item in values:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        name = str(item.get("name", ""))
        if args.number is not None and str(number) != str(args.number):
            continue
        if args.prefix and not str(number).startswith(str(args.prefix)):
            continue
        if args.name_contains and args.name_contains.lower() not in name.lower():
            continue
        matches.append(item)

    for item in matches[: args.limit]:
        print(
            json.dumps(
                {
                    "id": item.get("id"),
                    "number": item.get("number"),
                    "name": item.get("name"),
                    "ledgerType": item.get("ledgerType"),
                    "vatLocked": item.get("vatLocked"),
                },
                ensure_ascii=False,
            )
        )

    print(f"matches={len(matches)} total_accounts={len(values)} api_calls={client.calls_made}")
    return 0


def run_product_probe(client: TripletexClient, args: argparse.Namespace) -> int:
    _, payload = client.request(
        "POST",
        "/product",
        json_body={
            "name": args.name,
            "number": args.number,
            "priceExcludingVatCurrency": args.price,
            "vatType": {"id": args.vat_type_id},
        },
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"api_calls={client.calls_made}")
    return 0


def run_raw_request(client: TripletexClient, args: argparse.Namespace) -> int:
    params = json.loads(args.params) if args.params else None
    body = json.loads(args.body) if args.body else None
    _, payload = client.request(args.method, args.path, params=params, json_body=body)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"api_calls={client.calls_made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
