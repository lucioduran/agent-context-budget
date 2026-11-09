# Reference error analysis

No model response, answer alias or selection policy was revised after the held-out run.
The complete negative results remain in `results/evaluation.json`.

## Evidence and answer quality diverge

Full-window input returns ten correct numeric codes out of twelve but never the
required `[latest]` citation. On heldout-03 it emits `[least]`, an unknown citation
without an answer. On heldout-06 it emits the obsolete `6776` instead of `2849`.
Both facts were present, so this is not a retrieval failure.

Query selection retains every required fragment at both budgets. Nevertheless it
answers only six tasks correctly at 128 tokens and three at 256. More context does
not improve exact accuracy in this fixture/model combination. There is no statistical
or production claim from those small counts.

Relevant-sentence compression also retains every gold fragment, but exact accuracy
falls to one and two tasks at the two budgets. Outputs include `[note1]`, `[no]`,
`[2211]` and other bracketed strings rather than an answer with its source. The scorer
removes bracketed material before exact answer matching. A numeric code enclosed as
a citation does not satisfy the frozen output contract.

## What the ablations actually show

First-sentence compression drops all four buried corrections. For example, heldout-03
keeps `[latest] This record contains a routine administrative update.` while losing
`The current Cedar access code is 5926.` Citation-ID recall remains one, but evidence
recall falls to zero for that case. The aggregate retention is eight of twelve.

Removing the recency tie-break retains the same required evidence in every task.
The budgets allow both relevant records, so this dataset does not demonstrate a
retention advantage from that tie-break. Exact answer changes can reflect presentation
order and model behavior. The unit suite separately exercises a constrained budget
where the tie-break decides which one of two conflicting records survives.

## Design confounds and measurement limits

Selection policies present records in ranking order, while the prompt says later
records override earlier ones and does not print the original order field. That can
make chronology ambiguous after ranking: a stale record may follow its correction.
Consequently this experiment compares complete policy presentations, not an isolated
causal effect of information retention. A future separately frozen study should
render original chronological metadata or restore chronological presentation. We do
not retrofit that improvement to these results.

The pre-run prose mistakenly expected overflow past 512 tokens. Actual full inputs
fit the declared limit. The report explicitly has zero full-window truncations, and
no long-window model-quality claim is supported. Prefix truncation at the smaller
budgets is real and recorded. The overflow code path is covered by deterministic tests.

Resource totals have no repeated trials or warmup removal. The first model call has
initialization overhead, making comparisons against later policy calls especially
weak. RSS is a cumulative process high-water mark. There is no energy measurement,
human review study, interoperability certification or paid inference.
