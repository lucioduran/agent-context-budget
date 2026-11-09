"""Download only frozen model files to a project-local cache."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

manifest = json.loads(Path("data/model.json").read_text())
for name, expected in manifest["files"].items():
    path = hf_hub_download(
        manifest["model_id"], name, revision=manifest["revision"], local_dir=".cache/model"
    )
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Hash mismatch: {name}")
print("Pinned model files verified in .cache/model")
