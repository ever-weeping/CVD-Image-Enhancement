# CVD-Image-Enhancement

色觉异常(CVD)图像重着色 —— **自有方法的开发仓库**。2026-11-23 起接替
`code/` 先导实验,成为论文改进阶段的主线。

## 仓库三层结构

| 路径 | 内容 | 状态 |
|---|---|---|
| 根目录代码(`train.py` `evaluate.py` `data.py` `losses/` `models/` `metrics.py` `paths.py` `vendor/` `scripts/` `tests/` `docs/` `compat.py` `cvd_simulation.py` `SOURCE.json`) | 作者方法(Chen & Zhu et al., *Image recoloring for CVD compensation using Swin transformer*, NCAA 2024)的完整复现代码,自 `CVD-Swin-Reproduction`(只读冻结)导入 | 自有方法基座,继续开发 |
| `legacy_pilot/` | 2026-07~10 先导实验线(64px FiLM 条件重着色 + Machado 工具) | 冻结,不再开发 |
| `archive/pilot_runs_202607/` | pilot 论文统计产物(csv/json)+ `MANIFEST.sha256` | 只增不改 |

## 口径(约定)

- **author 口径**:作者原始实现(含 γ 空间 CVD 仿真怪癖),仅用于复现证据与消融对照。
- **linearized 口径(自有方法默认)**:仿真损失先转线性光;消融证明同增益下色度保真
  大幅改善(Pareto 改进),详见 `docs/METHOD_zh.md`。

## 同步与运行

- 本机(Mac)编辑 → `git push`;服务器 `~/PPWorkspace/CVD-Image-Enhancement` `git pull` 后训练。
- 数据与产物**永不进 git**:清单与图像在服务器 `~/mahai_ext/xiaolunwen_data/`
  (活跃清单 `cvd_fixed_v3.csv`),run 根 `~/mahai_ext/cvd_runs/`。
- test 集封存,方法定稿后一次性出分。
- 本仓库是 public:`.gitignore` 禁 `*.pdf *.png *.jpg`(论文原文与 COCO 派生图
  不可再分发),依赖与 provenance 只提交文本类产物。

## 复现状态(2026-11-23)

v2/v3 复现与线性化消融、α 权衡曲线全部 COMPLETE,判读见
`docs/AUTHOR_REPRODUCTION_zh.md` 与服务器 run 目录的 `REPORT_zh.md`。
