import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from context_budget.__main__ import main
from context_budget.core import parse_tasks, score, select
from context_budget.report import TraceTokenizer, digest, matrix, summarize, verify
from tests.test_core import Characters, fixture


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        self.raw = fixture()
        self.raw["split"] = "heldout"
        self.protocol = {"input_limit": 1000, "budgets": [250, 300], "max_new_tokens": 32}
        for name, value in [("heldout.json", self.raw), ("protocol.json", self.protocol)]:
            (self.root / "data" / name).write_text(json.dumps(value))
        hashes = {p.name: digest(p) for p in (self.root / "data").iterdir()}
        (self.root / "data/manifest.json").write_text(json.dumps(hashes))
        tok = TraceTokenizer(Characters())
        rows = []
        task = parse_tasks(self.raw)[0]
        for policy, budget in matrix(self.protocol):
            selected = select(task, policy, budget, 1000, tok)
            rows.append(
                {
                    "task_id": task.id,
                    "selection": selected.to_dict(),
                    "response": "9999 [new]",
                    "generated_ids": [1, 2],
                    "scores": score(task, selected, "9999 [new]"),
                    "selection_wall_seconds": 0.01,
                    "inference_wall_seconds": 0.02,
                    "inference_cpu_seconds": 0.03,
                    "peak_rss_bytes": 1234,
                }
            )
        self.report = {
            "schema": 1,
            "data_hashes": hashes,
            "raw_tasks": self.raw,
            "protocol": self.protocol,
            "tokenizer_trace": tok.trace,
            "rows": rows,
            "summary": summarize(rows),
        }

    def test_replay(self):
        self.assertEqual(
            verify(self.report, self.root), {"verified": True, "calls": 11, "groups": 11}
        )
        self.assertEqual(self.report["summary"]["full:1000"]["answer_exact"], 1)
        self.assertEqual(self.report["summary"]["full:1000"]["n"], 1)

    def test_tampering(self):
        mutations = [
            lambda r: r.update(schema=9),
            lambda r: r.update(data_hashes={}),
            lambda r: r["data_hashes"].update({"../escape": "bad"}),
            lambda r: r.update(raw_tasks={}),
            lambda r: r["rows"].pop(),
            lambda r: r["rows"].append(r["rows"][0]),
            lambda r: r.update(summary={}),
            lambda r: r["rows"][0]["selection"].update(prompt="changed"),
            lambda r: r["rows"][0].update(response=None),
            lambda r: r["rows"][0].update(response="wrong"),
            lambda r: r["rows"][0].update(generated_ids=[]),
            lambda r: r["rows"][0].update(generated_ids=[-1]),
            lambda r: r["rows"][0].update(generated_ids=[1] * 33),
            lambda r: r["rows"][0].update(inference_wall_seconds=float("nan")),
            lambda r: r["rows"][0].update(peak_rss_bytes=-1),
            lambda r: r["rows"][0].update(peak_rss_bytes="bad"),
            lambda r: r.pop("tokenizer_trace"),
        ]
        for mutation in mutations:
            report = copy.deepcopy(self.report)
            mutation(report)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(report, self.root)

    def test_file_changed(self):
        (self.root / "data/heldout.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify(self.report, self.root)

    def test_cli(self):
        path = self.root / "report.json"
        path.write_text(json.dumps(self.report))
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["verify", str(path), "--root", str(self.root)]), 0)
        self.assertTrue(json.loads(out.getvalue())["verified"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["verify", str(path / "absent")]), 2)
        with patch("context_budget.local.evaluate") as evaluate:
            self.assertEqual(main(["evaluate", "--model-path", "local", "--output", "result"]), 0)
            evaluate.assert_called_once()

    def test_existing_output_rejected_before_model_load(self):
        from context_budget.local import evaluate

        path = self.root / "existing.json"
        path.write_text("preserve")
        with self.assertRaisesRegex(ValueError, "already exists"):
            evaluate(Path("absent"), self.root, path)
        self.assertEqual(path.read_text(), "preserve")
