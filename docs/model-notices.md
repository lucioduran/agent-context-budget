# Model and dependency notices

The reference uses [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
at revision `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`, distributed under Apache-2.0.
Model weights are downloaded separately and not redistributed. The declared
512-token encoder limit comes from the tokenizer configuration at that pinned
revision. The model card documents training context and limitations; this project
does not revalidate upstream training claims.

Cached files were compared to the pinned Hugging Face revision: LFS SHA-256 for
large files and downloaded pinned bytes for other files. The resulting manifest is
committed. Evaluation verifies the same manifest before loading any weights.

PyTorch, Transformers, Hugging Face Hub, NumPy, SentencePiece and their transitive
dependencies retain their upstream licenses. `requirements-local.lock` records the
measured environment, not a redistribution of those packages.
