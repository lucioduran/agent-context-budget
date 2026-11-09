"""Budget policies and transparent evidence accounting."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

POLICIES = ("full", "head", "select", "compress", "select_no_recency", "compress_first")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    order: int


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    question: str
    documents: tuple[Document, ...]
    answers: tuple[str, ...]
    evidence: dict[str, str]


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_tasks(raw: Any) -> list[Task]:
    if not isinstance(raw, dict) or set(raw) != {"split", "tasks"}:
        raise ValueError("Expected split and tasks")
    if raw["split"] not in ("dev", "heldout") or not isinstance(raw["tasks"], list):
        raise ValueError("Invalid split or tasks")
    tasks = []
    seen = set()
    for item in raw["tasks"]:
        if not isinstance(item, dict) or set(item) != {
            "id", "category", "question", "documents", "answers", "evidence"
        }:
            raise ValueError("Invalid task fields")
        if not all(nonempty(item[k]) for k in ("id", "category", "question")):
            raise ValueError("Task strings must be nonempty")
        if item["id"] in seen:
            raise ValueError("Duplicate task ID")
        seen.add(item["id"])
        docs = []
        ids: set[str] = set()
        orders: set[int] = set()
        if not isinstance(item["documents"], list) or not item["documents"]:
            raise ValueError("Documents must be a nonempty list")
        for doc in item["documents"]:
            if not isinstance(doc, dict) or set(doc) != {"id", "text", "order"}:
                raise ValueError("Invalid document fields")
            if not isinstance(doc["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", doc["id"]):
                raise ValueError("Invalid citation ID")
            if not nonempty(doc["text"]) or type(doc["order"]) is not int:
                raise ValueError("Invalid document text or order")
            if doc["id"] in ids or doc["order"] in orders:
                raise ValueError("Duplicate document ID or order")
            ids.add(doc["id"])
            orders.add(doc["order"])
            docs.append(Document(**doc))
        answers = item["answers"]
        evidence = item["evidence"]
        if not isinstance(answers, list) or not answers or not all(map(nonempty, answers)):
            raise ValueError("Answers must be a nonempty string list")
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError("Evidence must be a nonempty mapping")
        for key, fragment in evidence.items():
            if key not in ids or not nonempty(fragment):
                raise ValueError("Unknown or empty evidence")
            if fragment not in next(d.text for d in docs if d.id == key):
                raise ValueError("Evidence must be a verbatim fragment")
        tasks.append(Task(item["id"], item["category"], item["question"],
                          tuple(sorted(docs, key=lambda d: d.order)), tuple(answers), evidence))
    if not tasks:
        raise ValueError("Tasks must not be empty")
    return tasks


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap(question: str, text: str) -> int:
    return len(words(question) & words(text))


def prompt(question: str, context: str) -> str:
    return (
        "Answer the question using the records. Later records override earlier records. "
        "Give only the answer followed by its citation in square brackets.\n"
        f"Question: {question}\nRecords:\n{context}\nAnswer:"
    )


@dataclass(frozen=True)
class Selection:
    policy: str
    budget: int
    prompt: str
    token_ids: list[int]
    original_tokens: int
    selected: list[dict[str, Any]]
    omitted_ids: list[str]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select(task: Task, policy: str, budget: int, limit: int, tok: Tokenizer) -> Selection:
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if type(budget) is not int or type(limit) is not int or not 0 < budget <= limit:
        raise ValueError("Budget must be positive and within the model input limit")
    cap = limit if policy == "full" else budget
    if len(tok.encode(prompt(task.question, ""))) > cap:
        raise ValueError("Budget cannot fit the question and instructions")
    docs = list(task.documents)
    if policy in ("select", "compress", "compress_first", "select_no_recency"):
        docs.sort(key=lambda d: (overlap(task.question, d.text),
                                d.order if policy != "select_no_recency" else -d.order),
                  reverse=True)
    candidates = []
    for doc in docs:
        text = doc.text
        if policy.startswith("compress"):
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if policy == "compress_first":
                text = sentences[0]
            else:
                text = max(enumerate(sentences),
                           key=lambda pair: (overlap(task.question, pair[1]), pair[0]))[1]
        candidates.append((doc, text))
    original = prompt(task.question, "\n".join(f"[{d.id}] {d.text}" for d in task.documents))
    selected: list[dict[str, Any]] = []

    def render() -> str:
        return "\n".join(f"[{s['id']}] {s['text']}" for s in selected)

    truncated = False
    for doc, text in candidates:
        selected.append({"id": doc.id, "text": text, "partial": False})
        if len(tok.encode(prompt(task.question, render()))) <= cap:
            continue
        selected.pop()
        truncated = True
        if policy not in ("full", "head"):
            continue
        tokens = tok.encode(text)
        # Exact re-encoding accounts for token boundary changes and prompt overhead.
        for length in range(min(len(tokens), cap), 0, -1):
            fragment = tok.decode(tokens[:length])
            selected.append({"id": doc.id, "text": fragment, "partial": True})
            if len(tok.encode(prompt(task.question, render()))) <= cap:
                break
            selected.pop()
        break
    final = prompt(task.question, render())
    included = {s["id"] for s in selected}
    omitted = [d.id for d in task.documents if d.id not in included]
    return Selection(policy, cap, final, tok.encode(final), len(tok.encode(original)),
                     selected, omitted, truncated or bool(omitted))


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score(task: Task, selection: Selection, response: str) -> dict[str, Any]:
    citations = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", response))
    answer = re.sub(r"\[[^\]]*\]", "", response)
    selected = {s["id"]: s["text"] for s in selection.selected}
    retained = [key for key, fragment in task.evidence.items()
                if key in selected and normalize(fragment) in normalize(selected[key])]
    lost = {key: fragment for key, fragment in task.evidence.items() if key not in retained}
    correct = normalize(answer) in {normalize(a) for a in task.answers}
    gold = set(task.evidence)
    return {
        "answer_exact": correct,
        "retained_evidence_recall": len(retained) / len(gold),
        "retained_citation_recall": len(gold & selected.keys()) / len(gold),
        "output_citation_recall": len(citations & gold) / len(gold),
        "output_citation_precision": len(citations & gold) / len(citations) if citations else 0.0,
        "grounded_exact": correct and gold <= citations and gold <= set(retained)
                          and citations <= gold,
        "lost_evidence": lost,
        "unknown_citations": sorted(citations - {d.id for d in task.documents}),
    }
