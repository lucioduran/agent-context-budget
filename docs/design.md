# Input, selection and evidence contracts

A dataset is a JSON object with exactly `split` (`dev` or `heldout`) and `tasks`.
Each task has `id`, `category`, `question`, `documents`, `answers`, and `evidence`.
IDs are unique per dataset. Each document has a unique citation `id` matching
`[A-Za-z0-9_-]+`, nonempty `text` and unique integer `order`. Higher order means
later; file order is normalized. No real timestamp or source trust is inferred.
Answers are a nonempty list of frozen accepted strings. Evidence maps required
citation IDs to nonempty verbatim fragments of those documents. Additional fields,
unknown evidence, duplicate orders and malformed values are rejected.

Budgets count the entire encoded prompt, including instructions, the question,
labels, suffix and special tokens. A budget that cannot fit the empty-context
prompt is an error. `full` uses the supplied model limit; other policies use the
supplied smaller budget. No tokenizer-level silent truncation is permitted.

Prefix policies append chronological records until a record does not fit, then
search descending token prefixes for a fragment whose complete prompt re-encodes
within budget. Later records are omitted. Selection/compression policies skip
non-fitting whole candidate records, then try remaining candidates. They never
partially keep a record. The `truncated` flag covers budget-induced omission or
partial retention. Extractive sentence removal is visible in selected text and
lost-evidence metrics even when this flag is false. `original_tokens` counts the
unmodified chronological prompt.

For relevant-sentence extraction, ties prefer the later sentence. Sentence splitting
uses punctuation followed by whitespace and is deliberately simple. There is no
special recognition of correction words, answer codes, gold labels or category.
Citation IDs accompany selected text and are never synthesized by compression.

A source ID can survive when its required fragment is lost. The scorer separates
these states, and grounded exact match requires the exact answer, all required
fragments retained, all gold citations output and no extra citations. Precision
uses unique citation IDs. Lost evidence is an explicit ID-to-fragment mapping.

The audit report retains tokenizer calls so CI can replay the same deterministic
selector without model libraries. Trace replay is not an independent token-count
audit. All matrix cells must occur exactly once; metrics and aggregate resource
sums must agree. Timings must be finite and nonnegative. Raw outputs may be wrong;
verification accepts consistent negative results. It rejects internally inconsistent
results, not poor model quality.

Model loading and SHA-256 validation precede inference. Generation is greedy with
one beam and no sampling. The decoder start token is removed before recording output
tokens; the generated EOS token remains in token counts but not decoded text. Encoder
and decoder budgets are separate for T5. There are no retries that hide failures.
