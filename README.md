# j-space 复现

在 Qwen3.5-0.8B 上本地复现 Anthropic 的 [A global workspace in language models](https://www.anthropic.com/research/global-workspace)(J-lens / j-space),并提供交互式可视化。

- 论文:[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- 官方参考实现:[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)(vendored 于 `vendor/`)
- 预拟合透镜权重:[neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)

## 使用

```bash
uv sync                                   # 安装依赖(含 vendor/jacobian-lens)
uv run python -m jspace_repro.run_readout # 重新计算 6 个演示 → viz/data/*.json
uv run python viz/build_page.py           # 组装静态页 → viz/index.html
uv run uvicorn app.server:app --port 8123 # 交互式服务(可输入任意 prompt)
uv run pytest                             # 测试(pursuit 单元测试 + 导出数据现象断言)
```

静态页 `viz/index.html` 直接双击打开即可(完全自包含)。

## 结构

- `jspace_repro/` — 核心:`core.py`(模型+透镜加载、读出计算)、`jspace.py`(非负稀疏分解)、`demos.py`(演示集)、`run_readout.py`(导出)
- `viz/` — `template.html` + `app.js`(静态页与 live 页共用),`build_page.py`,`data/`
- `app/server.py` — FastAPI:`POST /api/readout {prompt}` 实时读出
- `experiments/` — 筛选演示时的探针脚本(页面注记中排名数字的出处)
- `tests/` — pursuit 数学性质 + 导出数据必须支撑页面叙述的断言

## 与论文的已知偏差

页面"方法与忠实度"一节有完整说明:j-space 稀疏分解作用于传送后向量 J_ℓh(而非原始激活 h);pursuit 在 CPU 上运行(MPS 缺 `linalg_lstsq`);0.8B 规模下部分现象仅部分复现,均如实标注。

> vendored from [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) @ 581d398613e5602a5af361e1c34d3a92ea82ba8e (Apache-2.0)
