# j-space

Local reproduction of Anthropic's [A Global Workspace in Language Models](https://www.anthropic.com/research/global-workspace) (J-lens / j-space) on **Qwen3.5-0.8B**, with an interactive visualization.

**English** · [中文](#j-space-复现中文)

- Paper: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- Reference implementation: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (vendored in `vendor/`)
- Pre-fitted lens weights: [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)

## Where is the Qwen3.5-0.8B model?

**It is not stored in this repo.** It is downloaded from the Hugging Face Hub the first time you run anything, and cached on your machine — subsequent runs are offline.

| Artifact | Hugging Face source | Local cache location |
|---|---|---|
| Base model `Qwen/Qwen3.5-0.8B` (~1.6 GB) | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `$HF_HOME/hub/` (default `~/.cache/huggingface/hub/`) |
| Jacobian lens weights | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) → `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` | same cache |
| Token gloss (small, committed) | — | `vendor/jacobian-lens/assets/qwen_gloss.json.gz` |

The download is triggered by `transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)` and `JacobianLens.from_pretrained(LENS_REPO, ...)` in [`jspace_repro/core.py`](jspace_repro/core.py). To relocate the cache, set `HF_HOME` before running:

```bash
export HF_HOME=/path/with/space   # optional; where HF stores downloaded weights
```

Runs on Apple Silicon (MPS) or CPU — it auto-detects. No GPU required; the 0.8B model fits comfortably in memory.

## Run the experiment

Prerequisite: [uv](https://docs.astral.sh/uv/) (the project uses uv, not bare `pip`).

```bash
uv sync                                     # install deps, incl. vendored jlens
uv run python -m jspace_repro.run_readout   # recompute the 6 demos -> viz/data/*.json
uv run python viz/build_page.py             # assemble the static page -> viz/index.html
uv run pytest                               # tests: pursuit math + exported-data assertions
```

`uv sync` and the first `run_readout` will download the model + lens (see above); allow a few minutes on first run. `run_readout` writes one JSON payload per demo into `viz/data/` — the top-k token grid per (position, layer), tracked-token ranks, j-space sparse codes, and the model's greedy answer.

## Visualize it — static and dynamic

There are two front-ends, sharing the same `template.html` + `app.js`:

**1. Static page (offline, no server).** After `build_page.py`, just open the file:

```bash
open viz/index.html        # macOS; or double-click it in a file browser
```

It is fully self-contained (data baked in) — good for sharing a snapshot of the 6 reproduced phenomena.

**2. Dynamic / interactive server.** Run the FastAPI app; it loads the model + lens once and recomputes on demand:

```bash
uv run uvicorn app.server:app --port 8123
# then open http://localhost:8123
```

The live page is identical to the static one but adds an input box. Under the hood the browser calls `POST /api/readout {prompt}` ([`app/server.py`](app/server.py)); the server runs the model, applies the J-lens across layers, decomposes into j-space, and returns the readout, which the page renders live. **Type any prompt** and watch which concepts the lens surfaces in the middle layers — including "silent" intermediate answers that never appear in the input or output. Prompts are capped at 512 tokens; one readout runs at a time (MPS is not reentrant).

You can also hit the API directly:

```bash
curl -s localhost:8123/api/readout \
  -H 'content-type: application/json' \
  -d '{"prompt": "In the country shaped like a boot, what currency do they use?"}' | python -m json.tool
```

## Layout

- `jspace_repro/` — core: `core.py` (model + lens load, readout computation), `jspace.py` (non-negative sparse decomposition), `demos.py` (demo set), `run_readout.py` (export)
- `viz/` — `template.html` + `app.js` (shared by static and live pages), `build_page.py`, `data/`
- `app/server.py` — FastAPI: `POST /api/readout {prompt}` for live readouts
- `experiments/` — probe scripts used to select demos (source of the rank numbers in page annotations)
- `tests/` — pursuit math properties + assertions that exported data still supports the page's narrative

## Known deviations from the paper

Fully explained in the page's "Methods & faithfulness" section: the j-space sparse decomposition acts on the transported vector `J_ℓ·h` (not the raw activation `h`); pursuit runs on CPU (MPS lacks `linalg_lstsq`); at the 0.8B scale some phenomena reproduce only partially — all honestly labeled.

> Vendored from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) @ 581d398613e5602a5af361e1c34d3a92ea82ba8e (Apache-2.0)

---

# j-space 复现(中文)

[English](#j-space) · **中文**

在 Qwen3.5-0.8B 上本地复现 Anthropic 的 [A global workspace in language models](https://www.anthropic.com/research/global-workspace)(J-lens / j-space),并提供交互式可视化。

- 论文:[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- 官方参考实现:[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)(vendored 于 `vendor/`)
- 预拟合透镜权重:[neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)

## Qwen3.5-0.8B 模型在哪里?

**不在本仓库里。** 它在你第一次运行时从 Hugging Face Hub 下载,并缓存到本地——之后即可离线运行。

| 资源 | Hugging Face 来源 | 本地缓存位置 |
|---|---|---|
| 基础模型 `Qwen/Qwen3.5-0.8B`(约 1.6 GB) | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `$HF_HOME/hub/`(默认 `~/.cache/huggingface/hub/`) |
| Jacobian 透镜权重 | [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) → `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` | 同上缓存 |
| Token 释义表(小文件,已入库) | — | `vendor/jacobian-lens/assets/qwen_gloss.json.gz` |

下载由 [`jspace_repro/core.py`](jspace_repro/core.py) 中的 `transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)` 与 `JacobianLens.from_pretrained(LENS_REPO, ...)` 触发。若想改变缓存位置,运行前设置 `HF_HOME`:

```bash
export HF_HOME=/有空间的路径   # 可选;HF 存放下载权重的目录
```

可在 Apple Silicon(MPS)或 CPU 上运行,自动检测,无需 GPU;0.8B 模型内存占用很小。

## 运行实验

前置:[uv](https://docs.astral.sh/uv/)(本项目使用 uv,不用裸 `pip`)。

```bash
uv sync                                     # 安装依赖(含 vendor/jacobian-lens)
uv run python -m jspace_repro.run_readout   # 重新计算 6 个演示 → viz/data/*.json
uv run python viz/build_page.py             # 组装静态页 → viz/index.html
uv run pytest                               # 测试:pursuit 单元测试 + 导出数据现象断言
```

`uv sync` 与第一次 `run_readout` 会下载模型与透镜(见上),首次运行需几分钟。`run_readout` 为每个演示写出一份 JSON:每个(位置, 层)单元格的 top-k token 网格、被追踪 token 的排名、j-space 稀疏码,以及模型的贪心回答。

## 可视化——静态与动态

两个前端共用同一套 `template.html` + `app.js`:

**1. 静态页(离线,无需服务)。** 运行 `build_page.py` 后直接打开文件:

```bash
open viz/index.html        # macOS;或在文件管理器里双击
```

它完全自包含(数据已内联),适合分享 6 个已复现现象的快照。

**2. 动态 / 交互式服务。** 启动 FastAPI 应用,它只加载一次模型与透镜,按需实时计算:

```bash
uv run uvicorn app.server:app --port 8123
# 然后打开 http://localhost:8123
```

Live 页面与静态页一致,但多了一个输入框。背后浏览器调用 `POST /api/readout {prompt}`([`app/server.py`](app/server.py));服务端运行模型、逐层应用 J-lens、分解到 j-space,返回读出结果由页面实时渲染。**输入任意 prompt**,即可观察透镜在中间层浮现出哪些概念——包括从未出现在输入或输出中的"沉默"中间答案。prompt 上限 512 tokens;同一时刻只处理一个读出(MPS 不可重入)。

也可直接调用 API:

```bash
curl -s localhost:8123/api/readout \
  -H 'content-type: application/json' \
  -d '{"prompt": "In the country shaped like a boot, what currency do they use?"}' | python -m json.tool
```

## 结构

- `jspace_repro/` — 核心:`core.py`(模型+透镜加载、读出计算)、`jspace.py`(非负稀疏分解)、`demos.py`(演示集)、`run_readout.py`(导出)
- `viz/` — `template.html` + `app.js`(静态页与 live 页共用),`build_page.py`,`data/`
- `app/server.py` — FastAPI:`POST /api/readout {prompt}` 实时读出
- `experiments/` — 筛选演示时的探针脚本(页面注记中排名数字的出处)
- `tests/` — pursuit 数学性质 + 导出数据必须支撑页面叙述的断言

## 与论文的已知偏差

页面"方法与忠实度"一节有完整说明:j-space 稀疏分解作用于传送后向量 J_ℓh(而非原始激活 h);pursuit 在 CPU 上运行(MPS 缺 `linalg_lstsq`);0.8B 规模下部分现象仅部分复现,均如实标注。

> vendored from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) @ 581d398613e5602a5af361e1c34d3a92ea82ba8e (Apache-2.0)
