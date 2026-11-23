# baselines/ — 基线同集重测（统一口径对照实验）

执行方案与欠定台账：`docs/BASELINES_zh.md`（先读它，再动代码）。

- `run_eval.py` — 同集重测 harness，复用根目录 `metrics.py`/`data.py`/`cvd_simulation.py`，
  产物 schema 与 `evaluate.py` 完全一致，可直接并表。
- `methods/` — 每基线一模块，契约见 `methods/__init__.py` docstring；
  `identity.py` 是 harness 标定件（tcc_gain/SSIM/CD 应恰为 0/1/1）。
- `UPSTREAM.json` — 外部第三方码 provenance（url、pin commit、license 状态）。
- `_external/` — 上游 clone 与权重，**永不入 git**（.gitignore 已排除）。

运行示例（服务器，外置盘产物）：

```bash
python baselines/run_eval.py --method hassan19 --param beta=4 \
    --out ~/mahai_ext/cvd_runs/baselines_20261124/hassan19_beta4
python baselines/run_eval.py --method identity --limit 4 \
    --out ~/mahai_ext/cvd_runs/baselines_20261124/_harness_check
```

纪律：public 仓库不 commit 无 license 的第三方源码；对比表数字一律用我方重测值。

## 声明（DISCLOSURE）

本目录下的各方法实现是对**已发表方法的自行复现（re-implementations）**，仅用于
统一口径的对比评测。除注明外均为原创代码（从论文公式独立实现，或对本项目作者
自有复现代码的移植）；不含任何第三方源码。此处报告的数字代表**我方实现**在
v3 协议下的结果，不代表原作者发表的数值；各实现的复现假设与欠定决策逐项登记于
`docs/BASELINES_zh.md` 第 4 节。zhou24 模块内置的 `legacy_repro` 路径仅用于
移植忠实性验证，不产出任何论文数字。
