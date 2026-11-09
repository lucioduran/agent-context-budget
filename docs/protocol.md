# Frozen experiment protocol

The fixtures and this protocol are committed before any model evaluation. The two
optional development tasks have separate entities and codes. The twelve held-out
tasks are author-known, synthetic and template-related, not an externally blind
sample. No policy, prompt, labels or aliases will be tuned after held-out inference.

The fixed matrix has one full-window run per task at 512 tokens, plus five budgeted
policies at 128 and 256 tokens: 132 model calls. Greedy CPU float32 generation uses
32 decoder tokens and two threads. T5 is encoder-decoder: its declared tokenizer
input limit is 512; output tokens do not share the encoder budget. We enforce that
conservative input limit even though relative positions can technically extrapolate.

Policies use only the question and documents, never answers or evidence labels.
`full` consumes chronological records up to the declared input limit. `head` uses
the same prefix rule at the smaller budget. A final record can be partially kept;
the question, instructions, citation marker and answer suffix are always retained.
`select` sorts whole documents by unique query-word overlap then descending record
order, greedily packing only documents that fit. `compress` uses the same ranking,
then extracts one sentence per document by overlap, breaking ties toward the last
sentence. Original citation IDs stay attached. This is deterministic extractive
compression, not a learned or abstractive summarizer. `select_no_recency` reverses
the ranking tie-break. `compress_first` substitutes the first sentence. Those two
ablations expose losses from stale evidence and discarded corrections.

There are four tasks in each category: late evidence, conflicting old evidence,
and a correction buried in a later sentence. Each category has two short and two
long contexts. Long contexts exceed the input window; a bounded full-window result
must not be described as an untruncated oracle. This corpus only tests one required
evidence record per question and numeric code answers.

Exact answer match removes bracketed citations and normalizes case, whitespace and
punctuation. Retained citation recall measures surviving source IDs, while retained
evidence recall checks that the labeled verbatim fragment survives normalization.
Output citation precision/recall and grounded exact match are separate. Empty
citations score zero. These string metrics do not certify entailment or factuality.

Record full raw tasks, prompts, selected and omitted records, exact input token IDs,
raw responses, generated token IDs, lost gold fragments, per-call selection and
inference wall/CPU seconds, process peak RSS, and aggregate totals. RSS is a process
high-water mark including model loading, not per-call incremental allocation.
One observation per cell, no confidence interval or throughput claim. Model loading
is reported separately. No energy, electricity price or production cost inference.
All failures remain in the result artifact. Verification can replay deterministic
policy selection with the recorded tokenizer trace and rescore outputs without
loading the neural model; it cannot establish authenticity of inference.
