# legacy_pilot — 先导实验线(2026-07 ~ 2026-10),已冻结

本目录是仓库的**第一代研究线**:在 COCO val2017 64px 小子集上做的
FiLM 条件重着色先导实验与 Machado/Brettel 仿真工具。它与根目录的
作者方法复现/改进代码(`train.py` 等)**没有代码继承关系**,不要在此开发。

- 2026-11-23 整体移入并冻结(当时的 main 见 tag `repro-baseline-20261123` 之前的历史)。
- 论文统计依赖的冻结产物(csv/json)已归档到仓库根的
  `archive/pilot_runs_202607/`(含 `MANIFEST.sha256`),判读历史与 provenance
  修复说明见 `code/FROZEN_RUNS.md` 末节。
- 仅存本地磁盘、不入 git(public 仓库版权策略):`code/val2017/`(COCO 图像,
  787M)、`code/runs/*/run.log`、`code/runs/*/samples/*.png`、
  `Machado_simulate/` 下的论文 PDF 与示例/输出图像。备份责任在本人。
