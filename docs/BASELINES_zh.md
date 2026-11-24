# 基线同集重测执行方案（2026-11-23 开工拍板）

## 0. 定位

本目录下的工作即创新点候选③（统一指标口径重测全基线）的本体，同时为改进点①
（线性化 Pareto 改进）提供对比表。原则声明：**论文对比表中所有基线数字一律来自
我方同集重测**（v3 自然图 val + 三件套口径），原文表格数字只作趋势参照与定性旁证，
两套数字不交叉引用、不互相校验。

## 1. 名单与顺序

| 顺序 | 方法 | 谱系分支 | 实现方式 | 成本 | 状态 |
|---|---|---|---|---|---|
| 1 | **Gillooly25** (WACV25, cvd_vit) | 无训练逐图优化 (DINO 注意力) | 上游码 pin commit + 我方适配层 | 接入 ~1 人日 + GPU 逐图 200 步 ×1753×2轴×2版 | ⏸ 挂起，**是否纳入对比暂定**（用户 2026-11-24：待线下审上游无 license 码后再定） |
| 2 | **Hassan19** | 闭式/全局色度族 | 全自实现（无官方码） | 2–3 人日（估），CPU | ✅ 全量完成（β1–4×双光域×双轴 8 组合，2026-11-24 晨，见 §7） |
| 3 | **Zhou24** | B 快速硬约束族（Huang22 母体，替代 Zhu22） | **移植用户 2025-07 人工审过的复现码**（`/Volumes/sea_disk/颜色与视觉实验室/zhou_method/`，只读、不入 git）+ 协议化改造 | 大幅省于人日估算，CPU | ✅ 全量完成（双轴×双光域 4 路，2026-11-24 晨，见 §7；移植验收见 4.3） |
| — | **chen** (NCAA24) | C 学习 amortized | 复现 v3 已完成，数字直接进表 | 0 | ✅ 可用 |
| — | 我方 method (linearized α 族) | — | 已完成 | 0 | ✅ 可用 |

不纳入：Zhu22/Zhu19/Wang21/Huang22（退 related work，裁决见 mao-lab-genealogy 与各精读笔记）。

## 2. 评估协议（与自有方法完全同集同口径）

- 数据：`cvd_fixed_v3.csv`，source=natural，split=**validation**（1,753 张），256×256。
  test 继续封存。limit/seed 与 evaluate.py 同惯例（group_paths seed=2026 子集抽样）。
- 双轴 protan/deutan 各评一次；仿真器 **machado severity=1.0**（我方主口径），
  brettel 仅敏感性抽查（与 v3 复现评估口径一致）。
- 指标：只进 `metrics.batch_metrics`（TCC 在 CVD 模拟 Lab 上算、SSIM、CD λ=0；
  报告惯例 TCC增益/SSIM/色度保真=1−CD/252.7 三列，越高越好，联合判读）。
  **各基线自定指标（ΔL/ΔN/ΔC、E_nat/E_per、FSIMc、LCE）一律不用**——这正是
  "口径不可比"论证的落地方式；如需引用原文数字只做定性描述。
- 产物目录：服务器 `~/mahai_ext/cvd_runs/baselines_20261124/<method>_<参数串>/`，
  schema 与 evaluate.py 一致（per_image_metrics.csv / summary.csv / REPORT_zh.md
  表格行 `|natural_validation|...` 可被晨检 grep 直接汇总）。
- 每方法 × 每轴 × 每光域口径的档位都要单独出一个 run 目录，宁多勿混。

## 3. 验收标准（对"关键信息给不全"的制度化回答）

复现精度分三档，**只承诺前两档**：

- **A 不变量自检（硬性）**：实现内部与论文公式的结构性一致。例：
  Hassan β=1 ≡ Hassan17 公式路径（Ẑ=Z_o+Ê_x+Ê_z 且 Ê_x 同时进 X/Z）；
  Zhou24 α=0 ⇒ 输出色全部落在二色投影面内（simulate(output)≈output 的投影等价性）；
  Gillooly 关掉 offset 的 1×1 bias 级与正文消融方向一致（bias removal 影响最大）。
  另含数值健全性：输出 ∈[0,1]、指标有限、耗时符合量级预期。
- **B 趋势对齐（软性、必做）**：扫参单调性与相对排序对得上原文定性结论。例：
  Hassan E_nat(蓝放大)随 β 单调升、TCC 增益随 β 升；Zhou24 的 CD/LCE 随程度升;
  Gillooly 原始对比增益低于 Swin（我方 TCC 轴）而扰动保真占优（SSIM/CD 轴）。
  趋势也对不上时**先查实现**，查不出则如实记入欠定记录表，宁可少写一句 claim。
- **C 数值对齐（不承诺）**：原文表格数字与本重测数字之间的差距**不算失败**，
  每个差距归因到欠定记录表某一行。引用时永远用我方重测数字并注明参数档位；
  永远不写"复现了 XX 原文数字"。

## 4. 欠定记录表（每项 = 默认值 + 理由 + 敏感性方案 + 论文声明措辞）

### 4.1 Gillooly25（笔记 gillooly-vit-wacv2025）
| 欠定点 | 默认 | 理由 | 敏感性 | 声明 |
|---|---|---|---|---|
| 仿真矩阵作用光域 | γ 直通 [0,1]（其代码 ToTensor 后 einsum） | 忠实复现 | +线性化变体（我方口径，服务改进点①） | 双版并报 |
| 正文 Eq.2 与代码不一致 | 以代码为准 | 笔记裁决 | 不另做 | "实现跟代码不跟公式" |
| DINO 权重版本/patch | 上游 pin commit + dino.patch | 作者环境 | 不另做 | 记 commit hash |
| 其 Table 1 为跨类型跨 severity 平均 | 不可拆 | 单档明细在补充材料 | 我方单档自测代替 | 只引趋势 |
| 代码无 license | 不入库，外部 clone pin commit | public 仓库纪律 | — | setup 脚本+UPSTREAM.json |

### 4.2 Hassan19（笔记 hassan-flexible-contrast）
| 欠定点 | 默认 | 理由 | 敏感性 | 声明 |
|---|---|---|---|---|
| 光域（全文零语句） | 线性光 + Eq.1 原系数 | 色度学矩阵对 γ 值在物理上不自洽；线性光与我方消融口径一致 | γ 直通变体 | 双版并报，并入"全社区不指明光域"证据链 |
| Eq.1 矩阵系数（Travis1991 vs sRGB-D65） | 照抄论文 Eq.1 | 忠实复现优先 | 标准 sRGB-D65 变体 | 记差异 |
| 仿真器 Brettel97 vs Viénot99（端点） | brettel 双轴 | 我方后端现成、chen 系口径 | vienot 变体（protan 尤甚） | 记之 |
| 越界处理未写 | clip 到 [0,1] | chen"蓝饱和致对比损失"转述暗示 clip | 记录 clip 比例统计 | 声明 clip |
| β 档位（chen 表 β 未报） | β∈{1,2,3,4} 扫参 | 笔记建议两档起步；扫参可画曲线对 Pareto | — | 自报 β；不得写"chen 复测的是 Hassan17" |
| 主观仅 deutan、E_nat/E_per 色空间未写 | 不进主评 | 我方用三件套 | — | limitation 措辞照综述笔记 |
| 感知图归一化基准（原文"both sets normalized Y=100"，字面=各用自身 Y） | 各用自身 Y（字面直读） | 2026-11-24 回全文 p.1977 核对措辞；我方 brettel 后端非严格 Y-preserving，故 E 量受仿真 Y 漂移影响 | 后端换 vienot 已兼作检查（其矩阵更接近 Y-preserving） | 声明"各用自身 Y + 后端仿真器名" |

### 4.3 Zhou24（**主参照=用户复现码 zhou_method/，2026-11-24 用户提供**；笔记 zhou-fast-anomalous-vc2024 + huang-compensation-range-vc2022 为公式对照）
移植纪律：用户码里人工申过的自选一律照搬并挂账；我方只做三件事——协议化
（severity/仿真后端/Lab 后端/种子）、向量化（性能）、等价性对照（A 档锚点）。
| 欠定点 | 默认（=用户版除非注明） | 备注/协议化改动 | 声明 |
|---|---|---|---|
| Eq.7 方向符号规则 | a_c<0 翻 b_i；a_sim_g≥a_sim_c 翻 a_i | 用户申过，照搬 | 引用用户版规则 |
| 方向配对 | 每像素×1 全局随机像素（permutation，N 对）| 用户版**无邻域配对**（此前按 Wang21 转引链猜的 |P|=2N 作废，以代码为准）| 记"论文转引链与复现版不一致" |
| α 求解配对 | N 个全局随机 (i,j) 对、求和能量、3D Lab 含 L*、21 点共用同批对 | 用户版结构 | 同上 |
| k-means | k=6, random_state=0, n_init=10 | 原文无数值→用户申过 | — |
| 簇均值方向单位化 | **不**重归一（用户码算了 norm 未用） | 独立欠定点，挂账 | 声明 |
| LUT | 全 256³ 网格（非 262k 步长 4）；uint8 仿真色 Lab 为 key；精确匹配→KDTree 最近邻；候选取与原色 Lab 最近者（自然度优先）；反投影前 lab_mediate 过色域钳制 | 原文无数值→用户申过 | — |
| 仿真矩阵/光域 | 用户版=γ 域直通+两套**非 Machado 统一**矩阵（protan 硬编码 s0.4 档、deutan 为 Brettel 型混插）| **协议化改动①**：默认 sim_source='protocol'（我方 Machado s1.0，与评估/chen/Hassan 同后端）；'legacy_repro' 可整体复刻用户版 | 双轨并报，此为"结果对不上"主要来源 |
| 随机种子 | 用户版 permutation/配对**未固定种子**（仅 KMeans 有）| **协议化改动②**：我方固定 rng_seed=0 | 原版不可复现抖动，记欠定记录表 |
| Lab 后端 | 用户版=skimage | **协议化改动③**：统一 kornia（与 metrics.py 同后端），等价性对照实测数值差 | 记后端替换 |
| severity | 用户版演示 0.4 | 我方固定 1.0（v3 端点）| 对齐声明 |
| β 查表 | protan 0.3 / deutan 0.2 @1.0 | 与原文 Table 1 一致 | 引表 |

**移植验收结论（2026-11-24 凌晨，协调者独立复跑 selftest 与报告逐行一致）**：
- A 档等价性（legacy 轨 vs 用户原码，同种子同图）：方向场 5.3e-15、kmeans 平滑逐元素 0、
  α* 同格(-1.0)且能量曲线相对差 3.6e-16、反投影候选选择 0/255、端到端 mean 0.48/255
  （差值主因 jpg 回读与 Lab 后端，前向 rgb2lab 实测 ≤1e-13）。**移植忠实成立。**
- 新增/改判条目：① **簇均值方向不重归一**——用户码算了 norm 未用，两轨均照搬
  （我方"式 5 要求单位向量故应重归一"的分析意见**不进默认**，只留作审稿追问时的
  敏感性候选，防"复现夹带私货"质疑）；② 用户 build_lut 恒 protan@0.4 建表判为 bug，
  两轨均按实际轴/程度建表（0.4@protan 档仍与用户逐字节等价，已证）；③ legacy 轨在
  severity=1.0 用用户注释里的 protan 1.0 矩阵（用户本体恒 0.4，属协议化偏离，仅影响
  对照轨不影响主报 protocol 轨）；④ RNG 消费顺序复刻 permutation→randint×2；
  ⑤ 环境注：threadpoolctl dyld 崩溃仅进程内中和（不改数学）；KDTree 用 scipy，
  缺则退 sklearn；sklearn 版本差异可能改变簇分配（服务器环境固定后声明）；
  ⑥ 论文字面变体 zhou24_paper_literal.py **未跑通**（recolor 直接崩），只作阅读对照，
  不进任何 run。
- B 档唯一 FAIL：deutan 端点 9 色彩条 TCC 增益 −1.82（α*=−1 使红/绿两条相互塌缩；
  E 确实更低，方法属性非移植 bug）。自然图 kodim 双轴 +1.3/+2.1，原文对 deutan@100%
  亦仅称 "comparable"。**预期进表后 deutan 高程度档弱于 protan，写进 limitation 而非
  遮掩。**
- 耗时：256×256 warm 1.5–2 s/图，全量 1753×2 轴 ≈ 2 小时（单进程 CPU）。

## 5. 代码布局与 git 纪律

```
baselines/
├── run_eval.py          # 同集重测 harness（复用 metrics/data/cvd_simulation，与 evaluate.py 同 schema）
├── methods/
│   ├── __init__.py      # 动态注册：<name>.py 暴露 recolor(inp, axis, **params) + DEFAULTS + selftest()
│   ├── gillooly.py      # 适配层：调 baselines/_external 上游码
│   ├── hassan19.py      # 自实现
│   └── zhou24.py        # 自实现
├── UPSTREAM.json        # 外部码 provenance：url+commit+license 状态+获取命令
└── _external/           # 上游 clone + DINO 权重，.gitignore 排除，永不入库
```

纪律：public 仓库无 license 的第三方代码**不 commit**（Gillooly、DINO、cvd_matrices），
只 commit 适配层与 UPSTREAM.json；权重/数据/runs 一律外置盘或 _external；
欠定项改动先记欠定记录表再改代码。

## 6. 收尾清单（每个基线）

1. `recolor()` + `selftest()` 通过（A 档不变量，本地 CPU 小图）；
2. 服务器 200 图 smoke（--limit 200）看量级与趋势初判（B 档预检）；
3. 全量 val 1753×2轴×口径变体，产物进 baselines_20261124/；
4. 与 chen v3 行、我方 α 族行合并出 Pareto 对照图与表；
5. 欠定记录表回填（每个偏差归因），METHOD_zh 引用措辞红线复核
   （β≠α≠severity；Hassan 只引 Table 5 原数；Zhou24 不写"加速了 Zhu22"）。

## 7. 全量实测结果（2026-11-24 晨，v3 natural val 1753 图，评估侧统一 machado s1.0）

产物根：服务器 `~/mahai_ext/cvd_runs/baselines_20261124/`；合并汇总 `ALL_SUMMARY_20261124.txt`。
同集验证：所有 run 的 TCC输入列一致（protan 50.4331 / deutan 53.6312）。
格式：TCC增益 / SSIM / 色度保真（全部越高越好）。

| 方法（口径） | protan | deutan |
|---|---|---|
| Hassan19 β1 (γ) | −0.13 / 0.878 / 0.949 | +0.39 / 0.865 / 0.945 |
| Hassan19 β2 (γ) | +3.21 / 0.852 / 0.927 | +0.67 / 0.837 / 0.919 |
| Hassan19 β3 (γ) | +9.96 / 0.833 / 0.898 | +6.91 / 0.818 / 0.889 |
| Hassan19 β4 (γ) | +16.79 / 0.813 / 0.870 | +14.97 / 0.802 / 0.861 |
| Hassan19 β4 (linearized) | +14.84 / 0.779 / 0.864 | +2.80 / 0.781 / 0.874 |
| Zhou24 (γ) | +2.66 / 0.836 / 0.971 | +1.99 / 0.903 / 0.985 |
| Zhou24 (linearized) | +2.27 / 0.832 / 0.974 | +1.55 / 0.860 / 0.981 |
| chen 复现 (author 口径, v3) | +5.10 / 0.900 / 0.973 | +5.94 / 0.900 / 0.965 |
| 我方 α0.25 (linearized; protan=seed11) | +9.95 / 0.799 / 0.951 | +9.88 / 0.875 / 0.942 |
| 我方 α0.5 (linearized) | +5.02 / 0.905 / 0.980 | +6.03 / 0.907 / 0.980 |
| 我方 α0.75 (linearized) | +0.39 / 0.903 / 0.987 | +0.56 / 0.915 / 0.987 |

Hassan linearized β1–3 全表见 run 目录（deutan 低 β 负增益，光域敏感剧烈）；
β 扫参单调性（增益↑保真↓）四组合全部成立 → B 档趋势通过。

欠定记录表回填要点：
- **光域敏感性实测**：Hassan 剧烈（deutan β1–3 linearized 下增益 −6.4~−1.7 vs γ 下
  +0.4~+6.9，符号翻转）；Zhou24 温和（γ 比 linearized 增益高 +0.39/+0.44）。
  Zhou24 在其原生 γ 口径下更强 → 我方 linearized 主线不占基线便宜，公平性可辩护。
- **Pareto 结构**：保真 ≥0.94 区间增益前沿全部为我方 α 族（α0.25 双轴 +9.9）；
  Hassan 占低保真高增益角（+16.8@0.870），Zhou24 占高保真低增益角（0.985@+2.0，
  其保真极端值略高于我方 α0.5 的 0.980，如实记录）；chen≈我方 α0.5 但保真/SSIM 均低
  → 线性化 Pareto 改进在同表语境成立。
- **实测耗时**：Hassan 1–3 ms/图（全量 8 组合 <40 min）；Zhou24 服务器实测 8.2–8.4 s/图
  （单核；全量单路 ~4 h，四路并行一夜完成）。
- smoke(200图)→full(1753图) 数字一致性核验通过（Zhou24 protan +2.23→+2.27）。

**主表口径已拍板（2026-11-24，用户）**：基线各行取忠实口径（Hassan/Zhou24=γ、
chen=author），我方=linearized；光域敏感性全表进附录。理由见上"公平性可辩护"条。

## 8. 状态跟踪

- 2026-11-23：方案拍板开工；harness 与三个方法实现并行推进中。
- 2026-11-24：harness identity 标定 PASS；Hassan19 实现完成、自检通过（其间修正
  一处公式解读：感知图归一化按原文"both sets normalized (Y=100)"改为**各用自身 Y**，
  欠定记录表新增该行）；Gillooly 按用户决定挂起（外部无 license 码待线下审）；
  Zhou24 改走"移植用户复现码"主线，等价对拍+独立复跑验收通过（详见 4.3 验收结论；
  过程中两度出现 agent 报告与盘上实况不符，均靠复核拦截——纪律：报告只作线索，
  验收只看盘上复跑）。当晚审提交，随后服务器 smoke。
- 2026-11-24 晨：服务器 smoke 三路全过（identity 标定 0/1/1；Hassan/Zhou24 200 图趋势
  正常）→ 放行全量。Hassan 8 组合 + Zhou24 4 路全部 COMPLETE，合并表与判读见 §7。
  夜间异常仅本地 ssh 哨兵断线（远端 nohup 无恙）。
- 2026-11-24 用户拍板：① 主表口径=基线忠实口径+我方 linearized（已写入 §7）；
  ② zhou24_paper_literal.py 删除（本 commit 执行）；③ Gillooly 是否纳入对比暂定，
  继续挂起；④ 术语规范：弃用"台账"，统一称"欠定记录表"（用户偏好，全文档已替换）。
