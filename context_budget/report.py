"""Portable replay and aggregation of recorded local runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from context_budget.core import POLICIES, Tokenizer, parse_tasks, score, select


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TraceTokenizer:
    def __init__(self, backend: Tokenizer | None = None, trace: Any = None):
        self.backend = backend
        self.trace = trace if trace is not None else {"encode": {}, "decode": {}}

    def encode(self, text: str) -> list[int]:
        if self.backend is not None:
            self.trace["encode"][text] = self.backend.encode(text)
        result: list[int] = self.trace["encode"][text]
        return result

    def decode(self, tokens: list[int]) -> str:
        key = json.dumps(tokens)
        if self.backend is not None:
            self.trace["decode"][key] = self.backend.decode(tokens)
        result: str = self.trace["decode"][key]
        return result


def matrix(protocol: dict[str, Any]) -> list[tuple[str, int]]:
    return [("full", protocol["input_limit"])] + [
        (policy, budget)
        for budget in protocol["budgets"]
        for policy in POLICIES
        if policy != "full"
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"{row['selection']['policy']}:{row['selection']['budget']}"].append(row)
    result = {}
    for key, group in sorted(groups.items()):
        n = len(group)
        metrics = (
            "answer_exact",
            "retained_evidence_recall",
            "retained_citation_recall",
            "output_citation_recall",
            "output_citation_precision",
            "grounded_exact",
        )
        result[key] = {
            "n": n,
            **{metric: sum(r["scores"][metric] for r in group) / n for metric in metrics},
            "mean_input_tokens": sum(len(r["selection"]["token_ids"]) for r in group) / n,
            "mean_generated_tokens": sum(len(r["generated_ids"]) for r in group) / n,
            "truncated_calls": sum(r["selection"]["truncated"] for r in group),
            "selection_wall_seconds": sum(r["selection_wall_seconds"] for r in group),
            "inference_wall_seconds": sum(r["inference_wall_seconds"] for r in group),
            "inference_cpu_seconds": sum(r["inference_cpu_seconds"] for r in group),
        }
    return result


def verify(report: dict[str, Any], root: Path) -> dict[str, Any]:
    try:
        if report["schema"] != 1:
            raise ValueError("Unknown report schema")
        for name, expected in report["data_hashes"].items():
            if Path(name).name != name or digest(root / "data" / name) != expected:
                raise ValueError("Data hash mismatch")
        manifest = json.loads((root / "data/manifest.json").read_text())
        if report["data_hashes"] != manifest:
            raise ValueError("Incomplete data manifest")
        raw = json.loads((root / "data/heldout.json").read_text())
        protocol = json.loads((root / "data/protocol.json").read_text())
        if report["raw_tasks"] != raw or report["protocol"] != protocol:
            raise ValueError("Raw input mismatch")
        tasks = {task.id: task for task in parse_tasks(raw)}
        expected_cells = {
            (task_id, policy, budget) for task_id in tasks for policy, budget in matrix(protocol)
        }
        seen = set()
        tokenizer = TraceTokenizer(trace=report["tokenizer_trace"])
        for row in report["rows"]:
            selected = row["selection"]
            cell = (row["task_id"], selected["policy"], selected["budget"])
            if cell not in expected_cells or cell in seen:
                raise ValueError("Unexpected or duplicate matrix cell")
            seen.add(cell)
            task = tasks[row["task_id"]]
            replayed = select(
                task, selected["policy"], selected["budget"], protocol["input_limit"], tokenizer
            )
            if replayed.to_dict() != selected:
                raise ValueError("Selection replay mismatch")
            if not isinstance(row["response"], str):
                raise ValueError("Invalid response")
            if score(task, replayed, row["response"]) != row["scores"]:
                raise ValueError("Score mismatch")
            if (
                not isinstance(row["generated_ids"], list)
                or not all(type(t) is int and t >= 0 for t in row["generated_ids"])
                or not 0 < len(row["generated_ids"]) <= protocol["max_new_tokens"]
            ):
                raise ValueError("Invalid generated tokens")
            for key in (
                "selection_wall_seconds",
                "inference_wall_seconds",
                "inference_cpu_seconds",
                "peak_rss_bytes",
            ):
                value = row[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid resource measurement")
        if seen != expected_cells:
            raise ValueError("Incomplete matrix")
        if summarize(report["rows"]) != report["summary"]:
            raise ValueError("Summary mismatch")
        return {"verified": True, "calls": len(seen), "groups": len(report["summary"])}
    except (KeyError, TypeError, OSError, AttributeError) as exc:
        raise ValueError(f"Malformed or incomplete report: {exc}") from exc
