"""Command line: `python run.py` and `python validate.py` both land here."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .schema import InputError

ROOT = Path(__file__).resolve().parent.parent


def _parser(prog: str, desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=desc)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "eval.toml", help="TOML config file")
    p.add_argument("--profile", help="apply [profiles.<name>] from the config")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
    p.add_argument("--data-dir", help="shortcut for --set paths.data_dir=...")
    p.add_argument("--out-dir", help="shortcut for --set paths.out_dir=...")
    p.add_argument("--root", type=Path, default=ROOT, help="base for relative paths in the config")
    return p


def _config(args: argparse.Namespace):
    overrides = list(args.overrides)
    if args.data_dir:
        overrides.append(f"paths.data_dir={args.data_dir}")
    if args.out_dir:
        overrides.append(f"paths.out_dir={args.out_dir}")
    if getattr(args, "llm", None):
        overrides.append(f"llm.mode={args.llm}")
    return load_config(args.config, args.profile, overrides)


def run_main(argv: list[str] | None = None) -> int:
    from .pipeline import run_pipeline

    p = _parser("run.py", "Regenerate every artifact.")
    p.add_argument("--llm", choices=["auto", "live", "replay", "stub"], help="shortcut for --set llm.mode=...")
    p.add_argument("--offline", action="store_const", const="replay", dest="llm", help="same as --llm replay")
    args = p.parse_args(argv)
    try:
        result = run_pipeline(_config(args), args.root.resolve())
    except (ConfigError, InputError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"run {result['run_id']} stored in {result['storage']}: selected {result['selected']}, ready_to_ship={result['ready_to_ship']}")
    return 0


def validate_main(argv: list[str] | None = None) -> int:
    from .validate import validate

    args = _parser("validate.py", "Check every artifact.").parse_args(argv)
    try:
        errors = validate(_config(args), args.root.resolve())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("OK: artifacts present, valid and reproducible")
    return 0
