import copy
import json
import unittest
from pathlib import Path

from context_budget.core import POLICIES, parse_tasks, prompt, score, select

ROOT = Path(__file__).resolve().parents[1]


class Characters:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


def fixture():
    return {
        "split": "dev",
        "tasks": [
            {
                "id": "t",
                "category": "contradiction",
                "question": "current room code",
                "documents": [
                    {"id": "old", "order": 0, "text": "Current room code is 1111."},
                    {"id": "noise", "order": 1, "text": "Office furniture. " * 20},
                    {"id": "new", "order": 2, "text": "Routine update. Current room code is 9999."},
                ],
                "answers": ["9999"],
                "evidence": {"new": "Current room code is 9999."},
            }
        ],
    }


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.task = parse_tasks(fixture())[0]
        self.tok = Characters()
        self.base = len(self.tok.encode(prompt(self.task.question, "")))

    def test_fixture_split_and_categories(self):
        dev = parse_tasks(json.loads((ROOT / "data/dev.json").read_text()))
        held = parse_tasks(json.loads((ROOT / "data/heldout.json").read_text()))
        self.assertEqual(len(held), 12)
        self.assertFalse({t.question for t in dev} & {t.question for t in held})
        for category in {t.category for t in held}:
            self.assertEqual(sum(t.category == category for t in held), 4)

    def test_full_and_head_limits_and_partial(self):
        full = select(self.task, "full", self.base + 50, 1000, self.tok)
        self.assertFalse(full.truncated)
        self.assertIn("9999", full.prompt)
        head = select(self.task, "head", self.base + 65, 1000, self.tok)
        self.assertTrue(head.truncated)
        self.assertIn("new", head.omitted_ids)
        self.assertTrue(head.selected[-1]["partial"])
        self.assertLessEqual(len(head.token_ids), self.base + 65)
        self.assertTrue(head.prompt.endswith("Answer:"))
        self.assertIn(self.task.question, head.prompt)
        bounded = select(self.task, "full", self.base + 65, self.base + 65, self.tok)
        self.assertEqual(head.prompt, bounded.prompt)

    def test_exact_fit_and_zero_remaining(self):
        result = select(self.task, "head", self.base, 1000, self.tok)
        self.assertEqual(result.selected, [])
        budget = len(
            self.tok.encode(prompt(self.task.question, "[old] " + self.task.documents[0].text))
        )
        result = select(self.task, "head", budget, 1000, self.tok)
        self.assertEqual(len(result.selected), 1)
        self.assertFalse(result.selected[0]["partial"])

    def test_ranking_and_recency_ablation(self):
        cap = self.base + 52
        selected = select(self.task, "select", cap, 1000, self.tok)
        ablated = select(self.task, "select_no_recency", cap, 1000, self.tok)
        self.assertEqual(selected.selected[0]["id"], "new")
        self.assertEqual(ablated.selected[0]["id"], "old")
        self.assertEqual(score(self.task, selected, "9999 [new]")["retained_evidence_recall"], 1)
        self.assertEqual(score(self.task, ablated, "1111 [old]")["retained_evidence_recall"], 0)

    def test_compression_loses_correction_under_ablation(self):
        cap = self.base + 60
        good = select(self.task, "compress", cap, 1000, self.tok)
        bad = select(self.task, "compress_first", cap, 1000, self.tok)
        self.assertEqual(good.selected[0]["text"], "Current room code is 9999.")
        metrics = score(self.task, bad, "9999 [new]")
        self.assertEqual(metrics["retained_citation_recall"], 1)
        self.assertEqual(metrics["retained_evidence_recall"], 0)
        self.assertFalse(metrics["grounded_exact"])
        self.assertEqual(metrics["lost_evidence"], self.task.evidence)

    def test_labels_cannot_affect_selection(self):
        raw = fixture()
        raw["tasks"][0]["answers"] = ["1111"]
        raw["tasks"][0]["evidence"] = {"old": "1111"}
        other = parse_tasks(raw)[0]
        for policy in POLICIES:
            self.assertEqual(
                select(self.task, policy, self.base + 60, 1000, self.tok),
                select(other, policy, self.base + 60, 1000, self.tok),
            )

    def test_metrics_are_separate(self):
        selected = select(self.task, "full", 1000, 1000, self.tok)
        good = score(self.task, selected, "9999. [new]")
        self.assertTrue(good["answer_exact"])
        self.assertTrue(good["grounded_exact"])
        bare = score(self.task, selected, "9999")
        self.assertTrue(bare["answer_exact"])
        self.assertFalse(bare["grounded_exact"])
        self.assertEqual(bare["output_citation_precision"], 0)
        extra = score(self.task, selected, "9999 [new] [invented]")
        self.assertEqual(extra["output_citation_precision"], 0.5)
        self.assertEqual(extra["unknown_citations"], ["invented"])
        self.assertFalse(extra["grounded_exact"])
        self.assertFalse(score(self.task, selected, "The code is 9999 [new]")["answer_exact"])

    def test_multiple_required_sources(self):
        raw = fixture()
        raw["tasks"][0]["evidence"]["old"] = "1111"
        task = parse_tasks(raw)[0]
        selected = select(task, "select", self.base + 52, 1000, self.tok)
        self.assertEqual(score(task, selected, "9999 [new]")["retained_evidence_recall"], 0.5)

    def test_invalid_policy_and_budgets(self):
        for policy, budget, limit in [
            ("bad", 300, 500),
            ("head", 0, 500),
            ("head", 501, 500),
            ("head", True, 500),
            ("head", 10, 500),
            ("head", 300, 2.5),
        ]:
            with self.subTest(policy=policy, budget=budget), self.assertRaises(ValueError):
                select(self.task, policy, budget, limit, self.tok)

    def test_validation(self):
        cases = [
            [],
            {},
            {"split": "other", "tasks": []},
            {"split": "dev", "tasks": []},
            {"split": "dev", "tasks": [None]},
        ]
        changes = [
            ("id", ""),
            ("answers", []),
            ("answers", "code"),
            ("evidence", {}),
            ("evidence", {"missing": "x"}),
            ("evidence", {"new": "invented"}),
            ("documents", []),
        ]
        for key, value in changes:
            raw = fixture()
            raw["tasks"][0][key] = value
            cases.append(raw)
        for key, value in [("id", "bad[id]"), ("text", ""), ("order", True)]:
            raw = fixture()
            raw["tasks"][0]["documents"][0][key] = value
            cases.append(raw)
        for key in ("id", "order"):
            raw = fixture()
            raw["tasks"][0]["documents"][1][key] = raw["tasks"][0]["documents"][0][key]
            cases.append(raw)
        raw = fixture()
        raw["tasks"].append(copy.deepcopy(raw["tasks"][0]))
        cases.append(raw)
        raw = fixture()
        raw["tasks"][0]["documents"][0] = {}
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_tasks(raw)
