"""Fail-closed ICRA PDF length, anonymity, metadata, and font audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def run(*command: str) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if completed.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {completed.stderr.strip()}")
    return completed.stdout


def parse_info(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument(
        "--forbidden-identity",
        default="LinkeBai,github.com/LinkeBai,C:/Users/asus,damage-factorized-robot-arm",
    )
    args = parser.parse_args()
    info = parse_info(run("pdfinfo", str(args.pdf)))
    text = run("pdftotext", str(args.pdf), "-")
    fonts_output = run("pdffonts", str(args.pdf))
    source = args.source.read_text(encoding="utf-8")
    forbidden = [item.strip() for item in args.forbidden_identity.split(",") if item.strip()]

    errors: list[str] = []
    pages = int(info.get("Pages", "-1"))
    if pages < 1 or pages > args.max_pages:
        errors.append(f"page count {pages} is outside 1..{args.max_pages}")
    if info.get("Page size") != "612 x 792 pts (letter)":
        errors.append(f"unexpected page size: {info.get('Page size')}")
    if info.get("Encrypted", "").lower() != "no":
        errors.append("PDF must not be encrypted")
    for key in ("Author", "Title", "Subject", "Keywords"):
        value = info.get(key, "").strip()
        if value and value.lower() not in {"anonymous", "anonymous authors"}:
            errors.append(f"identifying or nonempty metadata {key}: {value!r}")
    if "Anonymous Authors" not in text:
        errors.append("rendered PDF does not contain Anonymous Authors")
    if not re.search(r"\\author\s*\{\s*Anonymous Authors\s*\}", source):
        errors.append("LaTeX source author is not exactly Anonymous Authors")
    combined = text + "\n" + source
    found_identity = [item for item in forbidden if item.lower() in combined.lower()]
    if found_identity:
        errors.append(f"forbidden identity strings found: {found_identity}")

    font_rows = []
    for line in fonts_output.splitlines()[2:]:
        columns = line.split()
        if len(columns) < 8:
            continue
        font_rows.append({
            "name": columns[0], "type": " ".join(columns[1:-6]),
            "embedded": columns[-5], "subset": columns[-4],
        })
    if not font_rows:
        errors.append("no font rows parsed")
    for font in font_rows:
        if font["embedded"].lower() != "yes":
            errors.append(f"font not embedded: {font['name']}")
        if font["type"] not in {"Type 1", "TrueType", "CID Type 0C", "CID TrueType"}:
            errors.append(f"unexpected font type: {font['name']} ({font['type']})")

    payload = {
        "status": "PASS" if not errors else "FAIL",
        "pdf": str(args.pdf),
        "source": str(args.source),
        "pages": pages,
        "max_pages": args.max_pages,
        "page_size": info.get("Page size"),
        "metadata": {key: info.get(key) for key in (
            "Author", "Title", "Subject", "Keywords", "Creator", "Producer"
        )},
        "anonymous_author_visible": "Anonymous Authors" in text,
        "forbidden_identity_strings": forbidden,
        "forbidden_identity_matches": found_identity,
        "fonts": font_rows,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
