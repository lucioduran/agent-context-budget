"""Command line entry points."""

import argparse
import json
import sys
from pathlib import Path

from context_budget.report import verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate and audit local context budgets")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("report", type=Path)
    check.add_argument("--root", type=Path, default=Path.cwd())
    run = commands.add_parser("evaluate")
    run.add_argument("--model-path", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            print(json.dumps(verify(json.loads(args.report.read_text()), args.root)))
        else:
            from context_budget.local import evaluate

            evaluate(args.model_path, args.root, args.output)
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
