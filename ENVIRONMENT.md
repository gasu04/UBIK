# UBIK Environment Setup

UBIK spans two nodes and four separate virtual environments, each with one
job. This replaces the old version of this file, which documented only the
Somatic node's legacy `~/pytorch_env` setup. Canonical reference: CLAUDE.md §3.5.

| Env | Node | Path | Python | Used by |
|---|---|---|---|---|
| DeepSeek venv | Hippocampal (macOS) | `/Volumes/990PRO 4T/DeepSeek/venv` | 3.13.7 | The DeepSeek RAG project (unrelated finance/LangChain app) — **not** used for UBIK code anymore |
| **UBIK venv (canonical)** | Hippocampal (macOS) | `/Volumes/990PRO 4T/UBIK/.venv` | 3.13.7 | `maestro`, `hippocampal`, `ingestion`, repo-root tooling |
| Somatic vLLM venv | Somatic (WSL2 Ubuntu) | `~/pytorch_env_vllm024` | 3.12.3 | vLLM inference server (active since the 2026-07-20 upgrade) |
| Somatic WhisperX venv | Somatic (WSL2 Ubuntu) | `~/ubik-whisperx-venv` | 3.12.3 | WhisperX transcription service |

`~/pytorch_env` (vLLM 0.13.0, the pre-upgrade venv) is retained on Somatic as
a rollback target but is not in active use.

---

## Hippocampal — UBIK venv (canonical)

```bash
source "/Volumes/990PRO 4T/UBIK/.venv/bin/activate"
# or, without activating:
alias maestro='cd "/Volumes/990PRO 4T/UBIK" && "/Volumes/990PRO 4T/UBIK/.venv/bin/python" -m maestro'
```

Built from the repo's own requirements (`requirements.txt` +
`maestro/requirements.txt` + `hippocampal/requirements.txt`), managed with
`uv` (no `pip` module installed in this venv — use `uv pip ...`).

Rebuild:
```bash
uv venv --python 3.13.7 .venv
uv pip install -r maestro/requirements.txt -r hippocampal/requirements.txt
```

Pinned versions actually installed: `requirements.lock` at the repo root
(regenerate with `uv pip freeze --python .venv/bin/python > requirements.lock`
after any dependency change — CLAUDE.md §2.7).

**Fixed 2026-07-27:** this venv previously had the full
torch/torchvision/torchaudio/transformers/trl/peft/accelerate/bitsandbytes/
tensorboard/wandb/jupyter training stack installed (258 packages), because
the top-level `requirements.txt` (a Somatic-flavored file, headed "PowerSpec
AI100 - RTX 5090 - CUDA 12.4") was included in the rebuild command. Rebuilt
from just `maestro/requirements.txt` + `hippocampal/requirements.txt` (155
packages) — `torch` itself remains as a genuine transitive dependency of
`sentence-transformers`, not leaked training-stack weight. See
`requirements.lock`'s header for the full note, including the still-open
question of `ingestion/requirements.txt`'s `openai-whisper` dependency.

---

## Somatic — vLLM venv

```bash
source ~/pytorch_env_vllm024/bin/activate
```

Installed versions (verified 2026-07-27):
- Python 3.12.3
- torch 2.11.0+cu129
- vllm 0.24.0+cu129
- GPU: NVIDIA GeForce RTX 5090, driver 591.86

Managed by `maestro` (`maestro/config.py::SomaticConfig.vllm_venv`, override
via `VLLM_VENV_PATH`) — the persistent `ubik-vllm` systemd unit
(HARDENING_PLAN_2026-07-23.md Layer A) invokes this venv's `python` directly.

Verify CUDA / GPU visibility:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Somatic — WhisperX venv

```bash
source ~/ubik-whisperx-venv/bin/activate
```

Python 3.12.3. Managed by `maestro`
(`maestro/config.py::SomaticConfig.whisperx_venv`, override via
`WHISPERX_VENV`).

---

## DeepSeek venv (legacy — do not use for UBIK)

`/Volumes/990PRO 4T/DeepSeek/venv` (Python 3.13.7) is a separate project's
venv — a junk-drawer of ~300 unrelated finance/LangChain/
opentelemetry-instrumentation-* packages. It was used to run `maestro` before
the UBIK venv existed (pre-2026-07-25); do not reintroduce that dependency.
See CLAUDE.md §3.5 for the full rationale.
