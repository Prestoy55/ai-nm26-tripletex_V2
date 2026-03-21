from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a local /solve request to the Tripletex agent using sandbox credentials."
    )
    parser.add_argument(
        "prompt",
        help="Task prompt to send to the local agent.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/solve",
        help="Local /solve URL. Defaults to http://127.0.0.1:8000/solve",
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
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Optional attachment path. Can be repeated.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional bearer token for the local service itself.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = [encode_file(Path(path)) for path in args.file]

    payload = {
        "prompt": args.prompt,
        "files": files,
        "tripletex_credentials": {
            "base_url": args.base_url,
            "session_token": args.session_token,
        },
    }

    headers = {"Content-Type": "application/json; charset=utf-8"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    response = requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=300,
    )

    print(f"HTTP {response.status_code}")
    if response.headers.get("content-type", "").startswith("application/json"):
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    else:
        print(response.text)

    return 0 if response.ok else 1


def encode_file(path: Path) -> dict[str, str]:
    mime_type = guess_mime_type(path)
    return {
        "filename": path.name,
        "mime_type": mime_type,
        "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def guess_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".txt":
        return "text/plain"
    if suffix == ".csv":
        return "text/csv"
    if suffix == ".json":
        return "application/json"
    return "application/octet-stream"


if __name__ == "__main__":
    raise SystemExit(main())
