# Agent Context Budget

[![CI](https://github.com/lucioduran/agent-context-budget/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/agent-context-budget/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Measure which evidence disappears when an agent's context shrinks, then test what
a pinned local model actually answers.

## Problem

A shorter prompt can discard a late correction while keeping an obsolete answer.
A compressed record can retain its citation label while losing the sentence that
justifies the answer. Token savings alone cannot show whether context is useful.

## Scope

Compare chronological full-window input, prefix truncation, query-based whole-record
selection and extractive sentence compression. Two ablations remove recency preference
and replace relevant-sentence extraction with first-sentence extraction.

This is an auditable local research CLI with twelve original fictional held-out tasks,
a frozen protocol and real FLAN-T5-small CPU inference. It is not a general-purpose
agent memory service. The portable policy and verification library has no runtime
dependencies. Neural evaluation requires the optional pinned model stack.

## Quick start

With Python 3.11 or newer, verify the published evidence without downloading a model:

```sh
git clone https://github.com/lucioduran/agent-context-budget.git
cd agent-context-budget
python -m context_budget verify results/evaluation.json
```

Verification replays selection from the recorded tokenizer trace, rescores every
response, checks the complete matrix and compares data hashes with the frozen files.
It does not rerun neural inference or prove the authenticity of timings.

## Usage

The Python API accepts a tokenizer exposing `encode(text)` and `decode(token_ids)`:

```python
import json
from pathlib import Path
from context_budget.core import parse_tasks, select, score
from context_budget.report import TraceTokenizer

report = json.loads(Path("results/evaluation.json").read_text())
task = parse_tasks(report["raw_tasks"])[0]
tokenizer = TraceTokenizer(trace=report["tokenizer_trace"])
selection = select(task, "compress", budget=128, limit=512, tok=tokenizer)
print(selection.prompt)
print(score(task, selection, "7391 [latest]"))
```

A trace can replay only strings encountered in its original run. For new tasks, use
a real tokenizer or implement the protocol. Labels are used solely for scoring;
selection reads only questions, text, citation IDs and explicit chronological order.
See [input and policy contracts](docs/design.md) and the [frozen protocol](docs/protocol.md).

CLI success exits with 0; invalid files, budgets or missing dependencies exit with 2.
The evaluator refuses existing output paths. Keep each experiment in a new artifact.

## Architecture

```text
Frozen tasks + model hashes + protocol
                  |
         rank / truncate / extract
                  |
  exact tokenizer budget with prompt overhead
                  |
     pinned offline encoder-decoder model
                  |
  raw response + evidence loss + resource measurements
                  |
        portable replay and aggregation
```

`core.py` contains validation, policies and metrics. `local.py` loads the local model
and records measurements. `report.py` replays deterministic selections and aggregates
scores. All inputs, selected fragments, omitted source IDs, prompts and raw outputs
are retained in the result artifact. Source hashes and the implementation commit
identify the code used for the measurement.

## Reproduction

The reference inference environment is Intel macOS, Python 3.11.13, PyTorch 2.2.2,
Transformers 4.46.3 and CPU float32. Use a project-local environment:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-local.lock
python scripts/download_model.py
python -m context_budget evaluate --model-path .cache/model --output build/new-run.json
python -m context_budget verify build/new-run.json
```

When running within the shared project batch, every heavy run must use the coordinator's
exclusive wrapper instead of invoking evaluation directly:

```sh
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python -m context_budget evaluate \
  --model-path .cache/model --output build/new-run.json
```

The recorded run used that wrapper and a read-only sibling model cache. Downloading is
separate from evaluation, which validates every model file and loads offline. The exact
model revision and SHA-256 hashes are in [data/model.json](data/model.json). The model
is not bundled. The complete package lock is specific to the measured platform; it is
not a claim that this old native stack installs on every new OS or Python version.

Fixtures were frozen in commit `1178a65` before inference. No prompt or policy tuning
followed held-out results. `scripts/freeze_tasks.py` explains fixture construction;
ordinary reproduction reads the committed files. See [protocol](docs/protocol.md).

## Results

Recorded on an Intel Xeon W-3223 at 3.50 GHz, 48 GiB RAM, macOS 26.7.1,
with two PyTorch 38.890 threads. There are 132 real model calls across eleven settings.

| Policy:input budget | Exact answer | Evidence retained | Citation IDs retained | Mean input tokens | Budget truncation | Total inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full:512 | 10/12 | 1.000 | 1.000 | 280.4 | 0/12 | 6.905 |
| head:128 | 0/12 | 0.000 | 0.000 | 128.0 | 12/12 | 1.461 |
| head:256 | 5/12 | 0.500 | 0.500 | 205.2 | 6/12 | 1.738 |
| select:128 | 6/12 | 1.000 | 1.000 | 111.4 | 12/12 | 1.225 |
| select:256 | 3/12 | 1.000 | 1.000 | 195.9 | 6/12 | 1.802 |
| compress:128 | 1/12 | 1.000 | 1.000 | 106.4 | 6/12 | 1.555 |
| compress:256 | 2/12 | 1.000 | 1.000 | 141.4 | 0/12 | 1.570 |
| select_no_recency:128 | 6/12 | 1.000 | 1.000 | 112.4 | 12/12 | 1.325 |
| select_no_recency:256 | 4/12 | 1.000 | 1.000 | 196.4 | 6/12 | 1.811 |
| compress_first:128 | 2/12 | 0.667 | 1.000 | 111.7 | 6/12 | 1.387 |
| compress_first:256 | 3/12 | 0.667 | 1.000 | 157.2 | 0/12 | 1.675 |

Every setting has **zero correct output citations and zero grounded exact matches**.
Keeping evidence did not guarantee that this small model used it correctly. First-sentence
compression dropped the required correction on all four buried-correction tasks while
retaining their citation IDs. Removing recency preference did not lose required evidence
at these budgets; that ablation is a negative result, not proof that recency is unimportant.

The original prompts contain 142 to 418 tokens. Contrary to the initial pre-run prose,
none exceeds the model's 512-token input limit. Full-window therefore has no truncation
in this run. Prefix truncation at 128 tokens loses every required late fact; at 256 it
loses six. The implementation has tested overflow handling, but this experiment does
not measure model behavior on prompts longer than its declared window.

Model loading took 1.068 seconds; peak process 1.196 was 1.196 GiB. Summed inference
wall time was 22.454 seconds and 38.890 time 38.890 seconds. Selection took 0.741 seconds
in total. These are single-run measurements, not throughput or latency estimates.
The first full-window call includes first-inference initialization. No warmup was excluded.

See the [complete raw report](results/evaluation.json), [summary](results/summary.json)
and [error analysis](docs/error-analysis.md). Results include all failures. Evidence replay
checks 132 cells; it does not certify production quality or independently authenticate
measurements.


## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover -v
python -m coverage report
python -m context_budget verify results/evaluation.json
```

CI uses Python 3.11.9 on Ubuntu 24.04, Intel macOS 15 and Windows Server 2022,
with actions pinned to verified commit IDs. Those interpreter builds were checked
against the [official manifest](https://github.com/actions/python-versions/blob/main/versions-manifest.json).
CI tests the portable implementation and recorded evidence, not neural inference.
Coverage gates the portable modules at 90%; the optional native model runner is
excluded and exercised by the recorded local evaluation.

## Limitations

- Twelve author-known synthetic tasks share templates. The split is frozen, not
  externally blind. Synthetic correctness is not evidence of production quality.
- The model is small and the task is limited to extracting numeric access codes.
  There is no multi-hop reasoning, long conversation state, real-user workload or
  human quality review. No evaluation data was used to train a model here.
- The model tokenizer declares a 512-token input limit. Full-window means bounded
  chronological input, not unlimited context. Long inputs are explicitly truncated.
- Compression is extractive and may omit corrections or qualifiers. It is not
  an evaluation of learned summarization. Selection uses lexical overlap, not semantics.
- String-based metrics miss valid answer paraphrases and do not establish entailment.
  Retaining a citation ID does not imply retaining the evidence associated with it.
- One observation per cell and fixed execution order do not support latency rankings,
  throughput claims or confidence intervals. Peak RSS includes loading and prior calls.
  Power consumption and hardware costs are unmeasured; no paid APIs were called.
- Replay trusts the recorded tokenization and does not prove model execution, timing
  authenticity or byte-identical behavior on other hardware. SHA-256 is not a signature.
- The CLI expects trusted local files, loads data in memory and is not a hardened
  upload service. Native model evaluation is supported on POSIX hosts; Windows CI
  covers portable policy and verification behavior only.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Add focused regression tests and separate
new experiments from the frozen reference. Preserve negative results and record
changes to prompts, data, policies or model versions.

## License

Code, original fictional fixtures, documentation and results use the [MIT license](LICENSE).
FLAN-T5-small is distributed separately under Apache-2.0. Dependencies retain their
own licenses. See [model notices](docs/model-notices.md).
