# CVD 感知条件重着色实验

本目录是独立 Git 仓库。原始图像、`runs/`、`artifacts/` 和 checkpoint
均不纳入 Git；正式结果由代码、运行目录中的 `config.json` 与
`split_manifest.csv` 共同复现。默认不保存 checkpoint，以避免占用 Mac
存储空间。

## 术语边界

实验接口使用“缺陷轴 + 模拟器控制量 `s`”：

- `protan, 0<s<1`：模拟 protanomaly（红异常三色视觉）条件；
- `deutan, 0<s<1`：模拟 deuteranomaly（绿异常三色视觉）条件；
- `protan, s=1`：模拟 protanopia（二色视觉端点）；
- `deutan, s=1`：模拟 deuteranopia（二色视觉端点）。

`s` 只是算法中的归一化模拟控制量，不是临床轻/中/重度、异常镜测量或
诊断标签。Machado 用于训练和主评价；Brettel 仅在 `s=1` 作第二模型
敏感性检查。二者属于相关的色觉建模谱系，因此该检查不是独立真实观察者
验证。

## 主要模块

- `cvd_simulation.py`：Machado、Brettel、Viénot 模拟和安全条件标签；
- `cvd_metrics.py`：输入固定混淆 mask、训练损失、逐图评价指标；
- `cvd_models.py`：容量匹配单流基线、三分支模型和消融；
- `cvd_recolor_experiments.py`：固定划分、多 seed 训练、压力集、配对
  bootstrap、Wilcoxon 与 Holm 校正；
- `scripts/make_paper_assets.py`：从冻结 CSV 生成论文图表；
- `tests/`：模拟、指标、模型以及 CPU–MPS 数值回归测试。

## 评价指标

- `cvd_contrast_gain`：输入固定 mask 内的模拟色度对比差值，越高越好；
- `cvd_contrast_ratio`：比值型补充指标，输入接近零时不作主指标；
- `delta_e00_mean` / `delta_e00_nonconf`：全图/非混淆区色差代价；
- `ssim_luma`：在 CIELAB L*/100 上计算的结构代理，不称为颜色自然度；
- `gamut_preclip_rate`：裁剪前越界率；
- 平均绝对 RGB 改变量、运行时间和可训练参数量。

这些都是指定模拟协议下的计算代理，不能替代真实 CVD 观察者的任务评价。

## 测试

```bash
/opt/anaconda3/bin/python3 -m unittest discover -s tests -v
```

## 论文所用冻结先导实验

```bash
/opt/anaconda3/bin/python3 cvd_recolor_experiments.py \
  --data val2017 \
  --stress-data ../dataset_gen/CVDdataset \
  --out runs/pilot_v2_20260720_fixed \
  --size 64 \
  --train-images 128 \
  --eval-images 24 \
  --stress-images-per-axis 12 \
  --batch-size 4 \
  --steps 40 \
  --width 12 \
  --seeds 7 19 \
  --train-simulator machado \
  --robustness-simulator brettel \
  --device mps \
  --sample-images 1 \
  --log-every 10 \
  --bootstrap-samples 2000
```

冻结主结果为：完整模型的描述性平均 `cvd_contrast_gain=0.124042`、
`delta_e00_mean=5.834555`；容量匹配基线为 `0.111595`、`6.012034`。
对比增益差异没有通过 Holm 校正，因此不能声称全面优于基线。

补充边界证据：

- `runs/condition_probe_20260720`：交换 FiLM 轴或固定 `s=0.5` 几乎不改变
  输出，当前短训练未建立可识别的显式条件机制；
- `runs/no_mask_ablation_20260720`：把 mask 输入、路由、融合和输出门控整链
  替换为全 1 后，对比增益升至 `0.239056`，但 ΔE00 升至 `11.865320`、
  SSIM-L* 降至 `0.861705`。这是同时改变残差幅度的复合干预，只支持
  系统级工作点变化，不能归因于某一个路由位置；
- 现有 `CVDdataset` 只作不参与调参的合成压力集，四个条件的主增益均为
  负，明确暴露了域外失败。

详细英文说明见 `EXPERIMENTS.md`。论文当前只把这些结果作为可复现先导
证据，不宣称 SOTA、临床个体化、真实观察者获益或 Pareto 前沿优势。

## 服务器实时进度与 TensorBoard

更新代码后，在训练环境运行 `python -m pip install -r requirements.txt`。
训练默认显示 tqdm 步数、速度、预计剩余时间和当前 loss；评价在交互终端
显示逐条件 batch 进度。`--no-progress` 可关闭进度条，原有
`--log-every` 文本日志仍保留。建议使用 `python -u` 配合 `tee` 保存日志。

TensorBoard 默认逐步记录总损失、结构/保真/CVD/解耦损失及学习率，
按模型与随机种子分别保存在 `<运行目录>/tensorboard/`，约每 10 秒刷新。
可用 `--no-tensorboard` 关闭。每次实验使用不同的 `--out` 目录，避免曲线混合。

服务器（在代码目录、已激活训练环境）：

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

本机另开终端建立 SSH 转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 -p 56887 mahai@101.7.187.88
```

随后浏览器打开 http://localhost:6006。旧实验不会自动出现曲线，
需要更新后重新启动训练。TensorBoard 不提供断线保活，长训练仍可在 tmux 中运行。
