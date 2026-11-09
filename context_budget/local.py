"""Optional offline CPU model adapter. Not imported by portable verification."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from context_budget.core import parse_tasks, score, select
from context_budget.report import TraceTokenizer, digest, matrix, summarize


class LocalTokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return cast(
            list[int], self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        )

    def decode(self, tokens: list[int]) -> str:
        return cast(str, self.tokenizer.decode(tokens, skip_special_tokens=True))


def evaluate(model_path: Path, root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Output already exists; preserve previous experiments")
    protocol = json.loads((root / "data/protocol.json").read_text())
    raw = json.loads((root / "data/heldout.json").read_text())
    tasks = parse_tasks(raw)
    model_manifest = json.loads((root / "data/model.json").read_text())
    if model_manifest["revision"] != protocol["revision"]:
        raise ValueError("Model revision mismatch")
    for name, expected in model_manifest["files"].items():
        if Path(name).name != name or digest(model_path / name) != expected:
            raise ValueError("Model file hash mismatch")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(protocol["threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import resource

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.model_max_length != protocol["input_limit"]:
        raise ValueError("Tokenizer limit differs from frozen model protocol")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.float32
    ).eval()
    load_seconds = time.perf_counter() - start
    tok = TraceTokenizer(LocalTokenizer(tokenizer))
    rows = []
    for task in tasks:
        for policy, budget in matrix(protocol):
            start = time.perf_counter()
            selection = select(task, policy, budget, protocol["input_limit"], tok)
            selection_seconds = time.perf_counter() - start
            inputs = torch.tensor([selection.token_ids], dtype=torch.long)
            start_cpu = time.process_time()
            start = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    max_new_tokens=protocol["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                )
            seconds = time.perf_counter() - start
            cpu_seconds = time.process_time() - start_cpu
            generated = result[0].tolist()[1:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rows.append(
                {
                    "task_id": task.id,
                    "category": task.category,
                    "selection": selection.to_dict(),
                    "response": response,
                    "generated_ids": generated,
                    "scores": score(task, selection, response),
                    "selection_wall_seconds": selection_seconds,
                    "inference_wall_seconds": seconds,
                    "inference_cpu_seconds": cpu_seconds,
                    "peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
                }
            )
        print(f"Completed {task.id}: {len(rows)} calls", flush=True)
    tracked_model = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "spiece.model",
        "model.safetensors",
    )
    report = {
        "schema": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_hashes": {
            str(p.relative_to(root)): digest(p)
            for p in sorted((root / "context_budget").glob("*.py"))
        },
        "data_hashes": json.loads((root / "data/manifest.json").read_text()),
        "model_hashes": {name: digest(model_path / name) for name in tracked_model},
        "raw_tasks": raw,
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "numpy", "sentencepiece")
            },
        },
        "model_load_seconds": load_seconds,
        "tokenizer_trace": tok.trace,
        "rows": rows,
        "summary": summarize(rows),
        "paid_api_calls": 0,
    }
    if platform.system() == "Darwin":
        report["environment"]["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        report["environment"]["physical_memory_bytes"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
