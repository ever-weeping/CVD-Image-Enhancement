# 作者 Swin 方法复现:协议与边界

代码位置:服务器 `~/PPWorkspace/CVD-Swin-Reproduction`。**自包含**:数据管线
(`data.py`)与 CVD 模拟(`cvd_simulation.py`)已内化,不再依赖同级
`CVD-Image-Enhancement` 项目。历史上的三分支改进与横向比较代码已备份至
`~/mahai_ext/archive/`,不属于本仓库。

## 来源

论文:Chen et al., *Image recoloring for color vision deficiency compensation
using Swin transformer*, Neural Computing and Applications 36, 6051–6066
(2024), DOI 10.1007/s00521-023-09367-2。
作者仓库:https://github.com/Ligeng-c/CVD_swin ,固定提交
`07271c5bc0f89068e8fc8cfa57cec0a9b8800549`。

`vendor/` 保留作者网络、训练及损失文件作为依据;只将网络的 timm 工具导入替换为
`compat.py`(所选网络 drop_path_rate=0)。实际使用作者 train.py 实例化的
`Generator_transformer_pathch4_844_48_3_nouplayer_server5`:patch=4、
embed_dim=96、encoder depths=[8,4,4]、window=8、三层 8 头、256×256 输入,
32,407,627 参数(含作者保留但未参与前向的层)。网络输出 Tanh,输入映射 [-1,1]、
输出映射回 [0,1]。

`losses/author.py` 整理作者损失:11×11 局部窗 120 个非中心偏移、3000 随机全局
像素对、L1 对比误差、SSIM 自然度;Adam lr=0.0002、betas=(0.5,0.999)。
`tests/verify_author.py` 核验其与作者原函数的损失及输入梯度一致性,并跑真实
256×256 GPU 训练步。

## 必须保留的区别(复现边界)

- **作者公开代码的适配复现**,不声称重新得到论文数值。源码将 Lab 各通道除以
  100/128/128 后,在再映射至 [-1,1] 的 Lab 上计算 SSIM;论文文字公式未完整说明
  该实现细节。训练保留此约定。
- 作者源码将 CVD 矩阵直接作用于编码 RGB 并截断,**未做 sRGB 线性化**;复现训练
  保留该行为。评估(`evaluate.py`)使用论文指标 TCC/SSIM/CD,模拟默认 Machado
  (含线性化),Brettel 作第二模型敏感性检查;不为作者模型更换训练约定。
- 原文 α=0 预训练后分 α=0.25/0.5/0.75 三支。默认协议每类型 20,000 步 α=0 +
  20,000 步 α=0.5,连续保留 Adam 状态;其余 α 用 `--alpha` 另跑。这是明确的预算
  选择,不是完整 120 轮或全分支复现。
- **α 分支协议**(`scripts/run_branch.sh`,`train.py --init-from`,2026-11-23 起):
  α=0.25/0.75 分支复用共享 α=0 预训练权重,在边界(step=pretrain_steps)恢复
  权重/优化器/采样 rng,只跑 finetune-steps,步号续写 20001–40000;config 记录
  init_from 路径与 sha256。同 seed、同数据顺序、同采样序列,α 是分支间唯一变量。
- **线性化消融**(`train.py --sim-train linearized`,`AuthorLoss(linearize=True)`,
  2026-11-23 用户决定):训练时模拟改为 sRGB 解码→Machado 矩阵→编码(与评估侧同
  口径),其余与默认 run 完全一致;默认 `author` 保留作者编码 RGB 约定。消融复用
  同一评估出口,报告与 v3 自然图行可直接并列对照。
- 每类型一个网络,不接收类型/程度条件;固定程度 1.0,不能宣称可控轻中度增强。
- 数据为重建固定清单(默认 `cvd_fixed_v3.csv`,含 SHA256 与 train/validation/test
  显式划分):自然图 34,953 张(31,453/1,753/1,747),按 ICCP12 Eq.15 邻域 CCPR
  < 0.8 筛选(方法、试点证据与偏差见数据目录 `FILTER_METHOD_zh.md`);合成图继承
  v1(作者网盘人工混淆色图)。**v3(2026-11-22 协议决定)**:合成图全部并入
  train(deutan 135,031 / protan 141,106),validation 与 test 只含自然图——与
  论文口径一致(§4.2 合成图仅辅助训练,§6.1 定量评估用真实照片);v2 仅把合成
  27,636 行划入 validation/test,已冻结,20261122 v2 run 经自身清单快照引用 v2。
  非作者原始 31,000 + 100,000 清单;每批 0.5 概率抽人工图。v1 冻结保留。
- 单 seed(默认 7)、固定步数预算;不足以作多 seed 显著性或论文优越性结论。
- 开发期只用 validation;test 为留出集,不做调参依据。

## 评估指标约定(metrics.py)

论文未完全指定的细节在此固定并记录:

- TCC(Eq.10):在 **CVD 模拟后**图像的 CIE Lab(kornia `rgb_to_lab`)上计算;
  全局 20,000 随机像素对 + 11×11 局部窗(边界取内部像素,与训练损失同惯例);
  每图采样种子由评估顺序索引决定,可复现。
- SSIM(Eq.6/7):RGB [0,1] 上 11×11 高斯 σ=1.5,c1=0.0001、c2=0.0009,逐通道均值。
- CD(Eq.11):λ=0,即 Lab 中 sqrt(Δa²+Δb²) 的逐图均值(重着色 vs 原图)。无天然
  [0,100] 区间:sRGB 色域内两色 a/b 位移硬上界为色度平面直径 252.7 Lab 单位
  (kornia 后端实测,`metrics.CD_SRGB_MAX`)。
- 展示惯例(2026-11-23 用户决定):色度保真 = 1 − CD/252.7 ∈ [0,1],越大越好,
  供图表与直观比较;单调线性映射,不改变任何比较方向,CD 原值仍是 Eq.11 报告量。
  作者论文 Table 3 的"CD(naturalness)"(0.9x、越大越好)与其 Eq.11 自相矛盾且
  未披露变换,我们不追随、也不与其数值对照。
- 判读:TCC 增益↑、SSIM↑、CD↓(或色度保真↑)需联合看;不同损失值之间不可横向比较。
- v3 下 validation/test 只含自然图,评估报告即自然图单组;对外主指标只报自然图,
  v2 run 报告中的 synthetic_validation 行属训练分布自检,不作为论文指标。

## 运行与输出

数据路径占位符在 `paths.py`;`read_manifest` 逐行解析并校验文件存在与哈希划分,
路径错误会立即报错。权重、日志、评估输出默认写 `~/mahai_ext/cvd_runs/`(外置盘),
不挤占 home 分区。

```bash
bash scripts/run_reproduction.sh          # deutan → protan,训练+validation 评估
cat ~/mahai_ext/cvd_runs/reproduction_<日期>/STATUS.txt
tail -f ~/mahai_ext/cvd_runs/reproduction_<日期>/train_deutan.log
```

TensorBoard:

```bash
/home/mahai/miniconda3/envs/image_enhance/bin/tensorboard \
  --logdir ~/mahai_ext/cvd_runs/reproduction_<日期> --host 127.0.0.1 --port 6006
# 本机转发:ssh -N -L 6006:127.0.0.1:6006 -p 56887 mahai@101.7.187.88
```

每 1000 步原子写 `latest.pt`(权重+优化器+步数+清单哈希),另存
`pretrained_alpha0.pt` 与 `final.pt`;`--resume` 恢复优化器与步数,数据排列从
seed 重启(非逐位重放)。运行目录拒绝覆盖已有运行。评估输出:
`per_image_metrics.csv`、`summary.csv`、`REPORT_zh.md`、`samples/`、`COMPLETE.json`。
