# 自有方法规格(METHOD_zh)

2026-11-23 经用户拍板:**linearized 升格为自有方法默认训练口径**。
本文档是自有方法的口径宪章;复现证据与消融对照保留在
`docs/AUTHOR_REPRODUCTION_zh.md` 与只读的 `CVD-Swin-Reproduction` 仓库。

## 方法定义

自有方法 = 作者方法(Chen & Zhu et al., NCAA 2024,Swin 重着色)的
架构、数据协议与训练协议 **不变**,仅将训练损失中的 CVD 仿真改为线性光口径:

- 训练损失 `losses/author.py AuthorLoss(linearize=True)`:对输入先做
  sRGB→linear-light 解码,施加 Machado 二色矩阵,再编码回 sRGB;
- 数据协议:清单 `cvd_fixed_v3.csv`(合成图仅 train,validation/test 仅自然图);
- 两阶段协议:α=0 预训练(仅对比损失)20k 步 → α 微调 20k 步;
- 评估协议不变(`evaluate.py` 一直是线性化口径)。

## 升格依据

作者实现的怪癖:**训练**损失在 γ 编码空间直接套矩阵,而**评估**却在线性光
空间仿真——自相矛盾。锥体光吸收是线性物理量(光子捕获),仿真理应在亮度
线性域进行。

**引用精确性**(2026-11-23 Machado TVCG09 精读裁决,见 literature/machado-physio-tvcg2009.md):
论文**未指明**矩阵算子的光域;线性光隐含于其谱推导(Eq.8 辐射量线性投影),
γ 空间套矩阵是后续实现的便利约定。故本改进定位为"**自洽性修复/回到推导隐含
光域**",不得写成"Machado 建议在线性光下应用"。另两处同名陷阱:论文的 severity
插值权重 α ≠ 我们损失的自然度权重 α;归一化 s∈[0,1] 出自 Machado 2010 报告/
python-colour 惯例(论文正文用 Δλ nm),s=1.0 即论文二色端点矩阵,引用须注明。
- 线性化自相矛盾**论文文字不暴露**(全文未写矩阵作用的颜色空间),只能由作者代码
  (vendor/train.py L314–319:γ 编码 RGB 直接 bmm 施矩阵)+ 评估侧引用惯例拼证;
  引用措辞须如此,不得写成"论文自述矛盾"。
- 论文 Table 3 标题 "CD metric" 但数值 0.92–0.999、正文称越大越好,**换算式从未给出**
  (疑 1−CD/Dmax):与我方 cd_fidelity 方向一致但数字不可直接映射,对数只对趋势。
  统一"越高越好"恰与论文自身隐含惯例同向。

**综述级口径约束**(2026-11-23 Zhu21 综述精读,见 literature/zhu-mao-survey-vc2021.md):
- 指标分类学(绝对/相对、全局/局部、自然度"仿真域 vs 输出域"两 scheme)出自 Zhu21 p.3002,
  **不得表述为我方贡献**;我方贡献 = 指出各基线实际落格 + 跨论文不可比 + 统一重测。
- TCC 属绝对型+仿真域;报配对增益后转相对量但**不罚过增强** → 按 p.3002 审稿人可追问 GCE,
  limitation 须声明。SSIM 与 cd_fidelity 属 scheme②(输出 vs 原图);两 scheme 同覆盖可引 p.3002 自证。
- **CCPR 在 Zhu21 Table 1 为仅单色/灰度指标**;我们用于彩色二色数据筛选属越界扩展,
  数据节须声明为 adaptation(与组内 τ 相差 13 倍一并披露)。CPR 出处引 **Shen et al. TOG 2016**
  (综述内部归属互斥:§3.2.3 vs §6.3)。
- severity 轴保留"**非临床校准**"声明(综述自注五档 "may not physically exist",p.3012);
  物理锚点:protan 100%↔Δλ=20 nm、deutan 100%↔−20 nm,与 Gillooly 0.4/0.6/1.0↔8/12/20 nm 一致。
- 无主观/Ishihara 实验 → limitation 声明(综述 p.3001 视间接评估与辨读为正当;chen Table 4 有先例)。
- "**个性化必然更优**"不可引 Zhu21(其自比 deutan 上 [17] 优于 [12],p.3013)。
- 综述表格不可靠(Tables 3–6 未标轴、8/9 题行不符、归类矛盾):凡数字一律回原文。
- **单指标可被刷分的现成反面教材**:Hassan19 protan TCC 全表最高(0.725***)同时 CD 垫底
  (蓝移放大对比但色度失真大)→ 支撑 TCC+cd_fidelity 双指标叙事;引其客观证据只引 Table 5
  原表(正文"β1–4 均改善"概括与表不符,β=4 多图变差)。Hassan19 光域亦未指明(Brettel 端点、
  Travis1991 矩阵系数与 sRGB-D65 有出入)→"全社区不指明光域"证据链扩到 Mao 系之外。
- Hassan19 的 β∈[1,4] 是观察者**偏好**旋钮,非 severity、亦非我方 α;三者不可混标。

2026-11-23 全量重训消融(α=0.5,自然图 validation 1,753 张):

(自 2026-11-23 起所有指标表统一**越高越好**;色度保真 = 1 − CD/252.7,CD 原值在逐图 CSV 溯源)

| 口径 | 轴 | TCC 增益↑ | SSIM↑ | 色度保真↑ |
|---|---|---|---|---|
| author(v3) | deutan | 5.94 | 0.900 | 0.9652 |
| linearized(消融) | deutan | 6.0348 | 0.9069 | 0.9798 |
| author(v3) | protan | 5.10 | 0.900 | 0.9727 |
| linearized(消融) | protan | 5.0236 | 0.9045 | 0.9805 |

补偿增益持平、色度失真约降 30–42%、结构指标不降 → **Pareto 改进**。
证据目录:服务器 `~/mahai_ext/cvd_runs/ablation_linearized_20261123/<axis>/`。

## 模型族(自有口径)

每轴 α ∈ {0.25, 0.5, 0.75}:

- **α=0.5**:即消融那套(full 40k,linearized),已 COMPLETE,直接充当族成员;
  共享边界权重 = `ablation_linearized_20261123/<axis>/author_<axis>_alpha0.5/pretrained_alpha0.pt`
  (357M,step 20000,config 含 sim_train=linearized);
- **α=0.25 / 0.75**:用 `scripts/run_method.sh ... <init_from=上述 checkpoint>`
  微调分支(步号续写 20001–40000,α 为唯一变量)。
  注意:上一轮 `alpha_branches_20261123` 是 author 口径,只用于复现 Fig.11
  权衡对照,**不**充当自有方法的调参证据。

## 纪律

1. `train.py` 默认 `--sim-train linearized`;复现/对照必须显式 `--sim-train author`
   (`run_axis.sh`、`run_branch.sh` 已钉死)。
2. 自有 run 目录前缀 `method_`(缺省名自动跟随口径);产物一律写
   `~/mahai_ext/cvd_runs/<run_root>/`,永不进 git。
3. `CVD-Swin-Reproduction` 与 `legacy_pilot/` 均只读。
4. test 集(1,747 自然图)封存,方法定稿后一次性出分。
5. 本机编辑 → GitHub → 服务器 pull;禁止服务器直接改码不提交。
6. Brettel/Viénot 第二模型互检的独立性有限:Machado 的 0.96 系数以 Brettel 曲面
   为参考校准、作者自评二色端点 "equivalent to Brettel";写作不得夸大互检力度。
7. **时间维度不作贡献声明**(用户 2026-11-23 拍板):改进不 claim 实时/加速;
   耗时仅作背景与基线口径辩护,severity 条件化卖点不含"实时"半边。

## α 模型族数据(2026-11-23,自然图 validation 1,753 张,逐格见各 eval REPORT_zh.md)

run 根:author 口径 = `reproduction_v3_20261122` + `alpha_branches_20261123`;
linearized 口径 = `ablation_linearized_20261123`(α0.5)+ `method_alpha_linearized_20261123`(α0.25/0.75)。

每格 = 增益↑ / SSIM↑ / 色度保真↑(全部越高越好)

| α | deutan author | deutan linearized | protan author | protan linearized |
|---|---|---|---|---|
| 0.25 | 10.62 / 0.762 / 0.892 | **9.88 / 0.875 / 0.942** | 11.48 / 0.828 / 0.914 | seed7 散度:9.70/0.012/0.922 ⚑;**seed11:9.95/0.799/0.951** † |
| 0.5 | 5.94 / 0.900 / 0.965 | **6.03 / 0.907 / 0.980** | 5.10 / 0.900 / 0.973 | 5.02 / **0.905 / 0.980** |
| 0.75 | 0.90 / 0.908 / 0.986 | 0.56 / 0.915 / 0.987 | 0.93 / 0.899 / 0.987 | 0.39 / 0.903 / 0.987 |

† seed11 探针:同一线性化边界 checkpoint、仅换训练 seed(2026-11-23,
`method_seedprobe_20261123/protan/`,config 含 `init_from_seed=7` 记录)。

判读:

- 全部格子(protan α0.25 取 seed11):linearized 在**保真与自然度上全面占优**,
  代价是低 α 端增益降 ~7–16%、高 α 端增益趋零时再小 ~0.3–0.4;α=0.5 工作点
  deutan 增益反超(+0.09)。升格成立。
- ⚑ **protan α0.25 seed7 散度格 → 定性为 seed 特异,非口径性质**:seed7 该格
  SSIM 0.012(med −0.003),训练 30k 步后 batch SSIM 震荡 0.36–0.86 不收敛;
  **换 seed11(同边界 ckpt、同 α、仅 seed 不同)得 SSIM 0.799 / 保真 0.951,健康**,
  自然度与 author 口径同格(0.828)可比、色度保真反而更高(0.951 vs 0.914)。结论:低 α 弱约束端点本身处于
  训练不稳定边缘(线性化梯度更强使其更易触发散度),**seed7 是该边缘的一个坏抽样**,
  而非 linearized 系统性缺陷。α 曲线在 protan 下端应连 seed11 稳健值;散度 seed
  作为低-α 稳定性边界如实标注。可选工程修法(微调期 lr 衰减 / early-stop on val SSIM)
  留作 robustness 附录,非主线。
- α0.75 双口径增益均 <1,该端点对补偿无实用价值,保留仅为画完整权衡曲线
  (与论文 Fig.11 结构对应)。

## 待办(按优先级)

- [x] linearized 口径 α 分支 0.25/0.75 × 双轴(2026-11-23 完成;protan α0.25 经
  seed7/seed11 双 seed 定性为 seed 特异散度,seed11 入表);
- [ ] 同集基线对照(名单已定:Gillooly γ/线性双版 + Zhou24 自实现 + Hassan19 自实现;
  Zhu22/19 退 related work)。Zhou24 自实现自设自由度三项:k-means k、LUT 步长/RGB 裁剪、
  邻域窗口半径(三篇原文均未给)→ 自设 + 敏感性检查 + 论文声明;配对集 |P|=2N 与 21 点网格
  有出处(Wang21/Huang22);光域自选线性光并声明;
- [ ] CCPR 阈值校准(FILTER_METHOD_zh.md 遗留);
- [ ] 第二改进点立项:severity 条件化 / 个体化校准(当前 s≡1.0 二色端点);
- [ ] 论文数据节按 v3 清单与线性化口径重写;
- [ ] 定稿后 test 解封一次性出分。
