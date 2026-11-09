import json
import re
import subprocess
import unittest
from pathlib import Path

from context_budget.report import digest, verify

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_frozen_evidence(self):
        report = json.loads((ROOT / "results/evaluation.json").read_text())
        self.assertEqual(verify(report, ROOT)["calls"], 132)
        self.assertEqual(json.loads((ROOT / "results/summary.json").read_text()), report["summary"])
        model = json.loads((ROOT / "data/model.json").read_text())
        self.assertEqual(report["model_hashes"], model["files"])
        self.assertEqual(report["protocol"]["revision"], model["revision"])
        for name, expected in report["source_hashes"].items():
            self.assertEqual(digest(ROOT / name), expected)
        self.assertTrue(
            all(
                len(row["selection"]["token_ids"]) <= row["selection"]["budget"]
                for row in report["rows"]
            )
        )

    def test_readme_headings(self):
        headings = re.findall(r"^## (.+)$", (ROOT / "README.md").read_text(), re.MULTILINE)
        self.assertEqual(
            headings,
            [
                "Problem",
                "Scope",
                "Quick start",
                "Usage",
                "Architecture",
                "Reproduction",
                "Results",
                "Testing",
                "Limitations",
                "Contributing",
                "License",
            ],
        )

    def test_text_convention(self):
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        for name in filter(None, paths):
            with self.subTest(path=name):
                self.assertNotIn(chr(0x2014), (ROOT / name).read_text(encoding="utf-8"))
