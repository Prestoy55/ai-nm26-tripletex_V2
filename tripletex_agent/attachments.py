from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from tripletex_agent.models import InboundFile, PreparedAttachment

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def materialize_attachments(files: list[InboundFile], attachment_dir: Path) -> list[PreparedAttachment]:
    attachment_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedAttachment] = []

    for index, inbound in enumerate(files):
        safe_name = f"{index:02d}_{sanitize_filename(inbound.filename)}"
        destination = attachment_dir / safe_name
        payload = inbound.decoded_bytes()
        destination.write_bytes(payload)

        prepared.append(
            PreparedAttachment(
                filename=inbound.filename,
                mime_type=inbound.mime_type,
                saved_path=str(destination),
                size_bytes=len(payload),
                extracted_text=extract_text(destination, inbound.mime_type),
            )
        )

    return prepared


def sanitize_filename(filename: str) -> str:
    sanitized = _UNSAFE_FILENAME.sub("_", Path(filename).name).strip("._")
    return sanitized or "attachment.bin"


def extract_text(path: Path, mime_type: str) -> str | None:
    if mime_type == "application/pdf":
        return extract_pdf_text(path)

    if mime_type.startswith("text/") or mime_type in {
        "application/json",
        "application/xml",
        "text/csv",
    }:
        return path.read_text(encoding="utf-8", errors="replace")

    return None


def extract_pdf_text(path: Path) -> str | None:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return None

    chunks: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            chunks.append(text)

    combined = "\n".join(chunks).strip()
    return combined or None
