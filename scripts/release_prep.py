#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _replace_one(pattern: str, repl: str, text: str, *, path: Path) -> str:
    updated, count = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"{path}: expected 1 replacement, got {count} for {pattern!r}")
    return updated


def set_cargo_version(cargo_toml: Path, version: str) -> None:
    text = cargo_toml.read_text(encoding="utf-8")
    text = _replace_one(
        r'^(version\s*=\s*)"[0-9A-Za-z.\-+]+"(\s*)$',
        rf'\g<1>"{version}"\g<2>',
        text,
        path=cargo_toml,
    )
    cargo_toml.write_text(text, encoding="utf-8")


def set_pyproject_version(pyproject_toml: Path, version: str) -> None:
    text = pyproject_toml.read_text(encoding="utf-8")
    text = _replace_one(
        r'^(version\s*=\s*)"[0-9A-Za-z.\-+]+"(\s*)$',
        rf'\g<1>"{version}"\g<2>',
        text,
        path=pyproject_toml,
    )
    pyproject_toml.write_text(text, encoding="utf-8")


def set_pyproject_torch_req(pyproject_toml: Path, torch_req: str) -> None:
    text = pyproject_toml.read_text(encoding="utf-8")
    patterns = [
        r'^(\s*)"torch\s*>=\s*[0-9.]+[^"]*"(\s*,?\s*)$',
        r'^(\s*)"torch\s*==\s*[0-9.]+[^"]*"(\s*,?\s*)$',
    ]
    replaced_any = False
    for pattern in patterns:
        updated, count = re.subn(
            pattern,
            rf'\g<1>"{torch_req}"\g<2>',
            text,
            flags=re.MULTILINE,
        )
        if count:
            replaced_any = True
            text = updated
    if not replaced_any:
        raise SystemExit(f"{pyproject_toml}: did not find a torch requirement to replace")
    pyproject_toml.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare repo versions for release builds.")
    parser.add_argument("--version", required=True, help="PEP 440 version (e.g. 0.0.1.290)")
    parser.add_argument(
        "--torch-req",
        default=None,
        help='Replace torch requirement string (e.g. "torch>=2.9,<2.10")',
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cargo_toml = root / "Cargo.toml"
    pyproject_toml = root / "pyproject.toml"

    set_cargo_version(cargo_toml, args.version)
    set_pyproject_version(pyproject_toml, args.version)
    if args.torch_req:
        set_pyproject_torch_req(pyproject_toml, args.torch_req)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

