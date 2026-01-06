#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _set_section_key_version(path: Path, section: str, key: str, version: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    in_section = False
    changed = False

    section_header = f"[{section}]"

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_header:
            in_section = True
            continue
        if in_section and stripped.startswith("[") and stripped.endswith("]"):
            in_section = False

        if not in_section:
            continue

        match = re.match(rf'^({re.escape(key)}\s*=\s*)"[^"]*"(.*)$', line)
        if not match:
            continue
        if changed:
            raise SystemExit(f"{path}: found multiple '{key} = ...' entries in [{section}]")
        lines[i] = f'{match.group(1)}"{version}"{match.group(2)}'
        changed = True

    if not changed:
        raise SystemExit(f"{path}: did not find '{key} = ...' in [{section}]")
    return "".join(lines)


def set_cargo_version(cargo_toml: Path, version: str, *, dry_run: bool) -> None:
    updated = _set_section_key_version(cargo_toml, "package", "version", version)
    if not dry_run:
        cargo_toml.write_text(updated, encoding="utf-8")


def set_pyproject_version(pyproject_toml: Path, version: str, *, dry_run: bool) -> None:
    updated = _set_section_key_version(pyproject_toml, "project", "version", version)
    if not dry_run:
        pyproject_toml.write_text(updated, encoding="utf-8")


def set_pyproject_torch_req(pyproject_toml: Path, torch_req: str, *, dry_run: bool) -> None:
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
    if not dry_run:
        pyproject_toml.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare repo versions for release builds.")
    parser.add_argument("--version", required=True, help="PEP 440 version (e.g. 0.0.1.290)")
    parser.add_argument(
        "--torch-req",
        default=None,
        help='Replace torch requirement string (e.g. "torch>=2.9,<2.10")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute changes but do not write files.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cargo_toml = root / "Cargo.toml"
    pyproject_toml = root / "pyproject.toml"

    set_cargo_version(cargo_toml, args.version, dry_run=args.dry_run)
    set_pyproject_version(pyproject_toml, args.version, dry_run=args.dry_run)
    if args.torch_req:
        set_pyproject_torch_req(pyproject_toml, args.torch_req, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
