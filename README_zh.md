# CVD Swin 作者方法纯复现

复现论文:Chen et al., *Image recoloring for color vision deficiency compensation
using Swin transformer*, Neural Computing and Applications 36, 6051–6066 (2024)。
作者源码:https://github.com/Ligeng-c/CVD_swin ,固定提交
`07271c5bc0f89068e8fc8cfa57cec0a9b8800549`(见 `SOURCE.json`,原文在 `vendor/`)。

本目录只包含**作者方法复现**:Swin U-Net 生成器 + 作者无监督损失
(Lab 局部/全局对比 + SSIM 自然度)。历史上的三分支改进、项目损失与横向比较
代码已移出(备份在服务器 `~/mahai_ext/archive/`),不再属于本仓库。
本仓库自包含,不依赖同级 `CVD-Image-Enhancement` 项目。

## 目录

|路径|职责|
|---|---|
|`train.py`|作者协议训练:α=0 预训练 → α 微调;Adam lr=2e-4、betas=(0.5,0.999)、batch 4、256×256|
|`evaluate.py`|论文指标评估:TCC(Eq.10)、SSIM(Eq.6/7)、CD λ=0(Eq.11)|
|`data.py`|固定清单读取、中心裁剪+ bicubic 数据集、seeded loader(内化自共享工具)|
|`metrics.py`|论文指标实现及其约定说明|
|`cvd_simulation.py`|Machado / Brettel / Viénot 可微 CVD 模拟(评估用)|
|`paths.py`|数据与运行目录配置(2026-11-22 切至 manifest v2:邻域 CCPR 自然图 34,953 张)|
|`models/author.py`|作者生成器入口(vendor 网络)|
|`losses/author.py`、`losses/ssim.py`|作者损失,数值约定与官方一致|
|`compat.py`|timm 最小兼容(所选网络 drop_path_rate=0)|
|`vendor/`|作者原始源码与环境 yaml,只读依据,不改网络|
|`tests/verify_author.py`|与作者原损失函数的数值/梯度一致性 + GPU 训练步检查|
|`scripts/run_reproduction.sh`|deutan/protan 训练+验证评估队列,输出写外置盘|
|`docs/AUTHOR_REPRODUCTION_zh.md`|复现边界与协议细节|

## 快速开始

```bash
# 1) 环境:沿用 image_enhance conda 环境,补依赖
/home/mahai/miniconda3/envs/image_enhance/bin/pip install -r requirements.txt

# 2) 数据路径校验(paths.py 已核实:自然图=places365_cvd_localccpr(v2 清单),
#    合成图根=xiaolunwen_data;read_manifest 逐行校验,路径错会直接报错)。
/home/mahai/miniconda3/envs/image_enhance/bin/python -c \
  "import paths; from data import read_manifest; print(len(read_manifest(paths.MANIFEST, paths.NATURAL_ROOT, paths.SYNTHETIC_ROOT)))"

# 3) 实现自检(需要空闲 GPU)
/home/mahai/miniconda3/envs/image_enhance/bin/python tests/verify_author.py

# 4) 训练+评估队列(默认 GPU0,CVD_GPU 可改;结果在 ~/mahai_ext/cvd_runs/)
bash scripts/run_reproduction.sh
```

单独训练/评估:

```bash
python train.py --axis deutan --alpha 0.5 --out ~/mahai_ext/cvd_runs/my_run
python evaluate.py --axis deutan --checkpoint ~/mahai_ext/cvd_runs/my_run/final.pt \
    --split validation --out ~/mahai_ext/cvd_runs/my_eval
```

`--resume` 恢复权重/优化器/步数(数据排列从 seed 重启,非逐位重放)。
运行目录拒绝覆盖:已有 `config.json` 时必须换新目录或 `--resume`。
开发期只用 `--split validation`;test 留出集不做调参依据。

## 复现边界(摘要)

- 作者公开代码的**适配复现**,不声称重现论文数值;数据为重建清单,非作者原始数据。
- 训练保留作者源码约定:CVD 矩阵直接作用于编码 RGB(无线性化);SSIM 在
  归一化 Lab 再映射 [-1,1] 上计算。评估改用论文指标 + Machado(含线性化),
  约定差异记录在 docs 中。
- 每类型一个网络、程度固定 1.0;α 默认 0.5,其余分支用 `--alpha 0.25/0.75` 另跑。
- 单 seed、固定步数预算(默认 20,000+20,000),不构成多 seed 显著性结论。
