# CLAUDE.md — j-space

Project facts for agents working in this repo. Behavioral house style lives in the workspace-level `CLAUDE.local.md`; this file is the encyclopedia (commands, layout, do-not-touch).

## What this is

Local reproduction of Anthropic's *A Global Workspace in Language Models* (J-lens / j-space) on **Qwen3.5-0.8B**, with a static and a live interactive visualization. Paper: <https://transformer-circuits.pub/2026/workspace/index.html>.

## Models & weights — where they live

Nothing model-sized is committed. Everything is pulled from Hugging Face on first run and cached under `$HF_HOME` (default `~/.cache/huggingface/hub/`):

- **Base model** `Qwen/Qwen3.5-0.8B` — `core.py:37`, via `transformers.AutoModelForCausalLM.from_pretrained`. ~1.6 GB.
- **Jacobian lens** `neuronpedia/jacobian-lens`, file `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` — `core.py:42`, via `JacobianLens.from_pretrained`.
- **Token gloss** `vendor/jacobian-lens/assets/qwen_gloss.json.gz` — committed (small).

The `jlens` package itself is **vendored** at `vendor/jacobian-lens/` (Apache-2.0, commit `581d398`) and wired in as an editable uv workspace member — do not `pip install jlens` over it.

Constants that pin all of the above: `jspace_repro/core.py` (`MODEL_ID`, `LENS_REPO`, `LENS_FILE`).

## Commands

```bash
uv sync                                     # install deps incl. vendored jlens
uv run python -m jspace_repro.run_readout   # recompute 6 demos -> viz/data/*.json
uv run python viz/build_page.py             # assemble static page -> viz/index.html
uv run uvicorn app.server:app --port 8123   # live server (type any prompt)
uv run pytest                               # pursuit unit tests + exported-data assertions
```

- Package manager is **uv** — never bare `pip`. Python 3.12 (`.python-version`).
- Device auto-detects: MPS on Apple Silicon, else CPU (`core.py:31`). Sparse pursuit is forced to CPU (MPS lacks `linalg_lstsq`); if you hit an MPS op gap elsewhere, `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Verify (three layers)

- **L1** — no separate linter/typechecker configured; keep imports clean.
- **L2** — `uv run pytest` (currently 18/18). Tests in `tests/` assert both pursuit math and that exported JSON still supports the page's narrative (`test_export.py`) — a passing test encodes the claimed phenomenon, so don't loosen an assertion to make it green.
- **L3** — static page renders (`viz/index.html`, self-contained) and/or the live server answers `POST /api/readout`.

## Layout

- `jspace_repro/` — core package. `core.py` (load model+lens, `compute_readout` payload), `jspace.py` (non-negative sparse decomposition / pursuit), `demos.py` (6 bilingual demos), `run_readout.py` (export).
- `viz/` — `template.html` + `app.js` (shared by static and live pages), `build_page.py`, `data/*.json` (precomputed demo payloads).
- `app/server.py` — FastAPI. `GET /` serves the live page; `POST /api/readout {prompt, chat}` recomputes for an arbitrary prompt. One readout at a time (global lock; MPS not reentrant). `MAX_PROMPT_TOKENS = 512`.
- `experiments/` — probe scripts used to *select* demos and source the rank numbers quoted in page annotations. Reference material, not part of the pipeline.
- `tests/` — `test_jspace.py`, `test_export.py`.
- `vendor/jacobian-lens/` — vendored upstream; treat as read-only.

## Known deviations from the paper

Documented in the page's "Methods & faithfulness" section: j-space sparse decomposition acts on the *transported* vector `J_ℓ·h` (not raw activation `h`); pursuit runs on CPU; at 0.8B some phenomena reproduce only partially (all honestly labeled in the demo notes).

## Do not touch

- `vendor/` — upstream code; changes belong in `jspace_repro/`.
- `.venv/`, `uv.lock` — managed by uv (`uv.lock` only via `uv` commands).
- Test assertions that pin a phenomenon's rank — those are the reproduction's evidence, not incidental values.
