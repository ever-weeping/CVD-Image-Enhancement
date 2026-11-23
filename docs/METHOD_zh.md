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
线性域进行。2026-11-23 全量重训消融(α=0.5,自然图 validation 1,753 张):

| 口径 | 轴 | TCC 增益 | CD | 色度保真 1−CD/252.7 | SSIM |
|---|---|---|---|---|---|
| author(v3) | deutan | 5.94 | 8.78 | 0.965 | 0.900 |
| linearized(消融) | deutan | 6.03 | 5.10 | 0.980 | 略升(见 run 目录 REPORT) |
| author(v3) | protan | 5.10 | 6.87 | 0.973 | 0.900 |
| linearized(消融) | protan | 5.02 | 4.94 | 0.980 | 略升(见 run 目录 REPORT) |

补偿增益持平、色度失真约降 30–42%、结构指标不降 → **Pareto 改进**。
证据目录:服务器 `~/mahai_ext/cvd_runs/ablation_linearized_20261123/<axis>/`。

## 模型族(自有口径)

每轴 α ∈ {0.25, 0.5, 0.75}:

- **α=0.5**:即消融那套(full 40k,linearized),已 COMPLETE,直接充当族成员;
  其 `method_/author_<axis>_alpha0.5/pretrained_alpha0.pt` 是线性化族共享边界权重;
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

## 待办(按优先级)

- [ ] linearized 口径 α 分支 0.25/0.75 × 双轴(4 分支 ≈ 一晚);
- [ ] 同集基线对照(Hassan 全局 / Zhu 原法,至少指标同集可比);
- [ ] CCPR 阈值校准(FILTER_METHOD_zh.md 遗留);
- [ ] 第二改进点立项:severity 条件化 / 个体化校准(当前 s≡1.0 二色端点);
- [ ] 论文数据节按 v3 清单与线性化口径重写;
- [ ] 定稿后 test 解封一次性出分。
