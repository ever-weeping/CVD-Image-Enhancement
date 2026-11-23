# 固定划分与自然图／人工图混合训练

此流程使用我们重建的 Places365 子集，不代表作者原始 31,000 张自然图或论文训练方案的复现。自然图已有 train=10,000、validation=1,000、test=1,000，直接保留，不随实验 seed 重新划分。

## 1. 服务器生成一次固定清单

同步代码后，在服务器执行：

```bash
cd /home/mahai/PPWorkspace/CVD-Image-Enhancement/code
conda activate image_enhance
DATA_ROOT="$HOME/mahai_ext/xiaolunwen_data"
python scripts/prepare_fixed_manifest.py \
  --natural-root "$DATA_ROOT/places365_cvd_ranked_12000" \
  --synthetic-root "$DATA_ROOT" \
  --output "$DATA_ROOT/cvd_fixed_v1.csv" \
  --seed 20261010 --workers 4
```

人工图读取 `Color_cvd_P_experiment_100000` 和 `Color_cvd_D_experiment_100000`。脚本计算实际文件 SHA256，按内容哈希确定约 90%/5%/5% 的训练／验证／测试划分，并全局去除字节完全相同的文件（自然图优先，人工图 P 优先于 D）。自然图若存在跨划分重复则报错；不会做感知近重复检测。每 512 个文件显示进度，实际数量见输出的 `.meta.json`。

清单拒绝覆盖。之后的所有实验复用同一清单；不要修改清单对应的图片。训练读取时检查路径、清单内的哈希重复和划分，出于启动效率不会重新计算全部图片哈希。

## 2. 最小混合训练

```bash
python cvd_recolor_experiments.py \
  --fixed-manifest "$DATA_ROOT/cvd_fixed_v1.csv" \
  --data "$DATA_ROOT/places365_cvd_ranked_12000" \
  --synthetic-root "$DATA_ROOT" --synthetic-ratio 0.5 \
  --variants baseline ours_full --seeds 7 \
  --axes protan deutan --train-severities 0.25 0.5 0.75 1.0 \
  --steps 100 --size 96 --width 16 --batch-size 4 \
  --fixed-eval-limit 16 --num-workers 4 --device cuda \
  --save-checkpoints --out runs/fixed_mixed_smoke
```

每一步先抽取类型和程度，再以 0.5 概率抽取人工图批次，否则抽自然图。P 图仅用于 protan，D 图仅用于 deutan；程度仍按训练参数抽取。0.5 是本项目可调的采样策略，不是作者论文比例，也不保证短训练恰好各半。不同模型使用相同 seed 时共享相同条件与数据来源抽样序列。人工图作为输入参与现有损失训练，没有新增成对目标图监督。

`--fixed-eval-limit 16` 是每个验证组最多 16 张，仅用于快速检查。正式验证使用 `--fixed-eval-limit 0`，读取全部验证图；自然、P 人工、D 人工分别报告指标。`--train-images` 和 `--eval-images` 在固定清单模式下无效；不能同时使用旧 `--stress-data`。

正式训练可调整步数及输出目录，例如 `--steps 80000 --fixed-eval-limit 0 --out runs/fixed_mixed_80000`。可用 `--checkpoint-every 1000` 每1000步保存最新中间权重；这不包含优化器状态，不提供完整断点续训。输出目录不能复用已有实验目录。

自然图单独训练对照保留同一清单，只改 `--synthetic-ratio 0`，仍在相同自然与人工留出集上评估。这里的 baseline 仍是项目现有单流 CNN 对照，不是作者 Swin 方法的复现。

## 3. 最终测试

调参使用 validation。最终 test 必须单独加载已训练检查点，例如：

```bash
python cvd_recolor_experiments.py \
  --fixed-manifest "$DATA_ROOT/cvd_fixed_v1.csv" \
  --data "$DATA_ROOT/places365_cvd_ranked_12000" \
  --synthetic-root "$DATA_ROOT" \
  --variants ours_full --seeds 7 --width 16 --size 96 \
  --fixed-eval-split test --fixed-eval-limit 0 \
  --eval-only-checkpoint runs/fixed_mixed_80000/checkpoints/ours_full_seed7.pt \
  --batch-size 4 --num-workers 4 --device cuda \
  --out runs/fixed_mixed_80000_test
```

检查点必须匹配清单 SHA256、variant、seed 和 width。评估尺寸和条件应与预定实验协议保持一致。一次只评估一个检查点；测试命令不会更新权重。

## 4. 查看记录

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

在本机另开终端建立转发，再访问 `http://localhost:6006`：

```bash
ssh -N -L 6006:127.0.0.1:6006 -p 56887 mahai@101.7.187.88
```

- `tensorboard/`：损失和 `train/synthetic_batch`（1=人工，0=自然）。
- `sampling_*_seed*.json`：混合训练各来源、类型实际批次数与图片次数。
- `fixed_manifest.csv`、`config.json`：冻结清单副本、哈希、数量和参数。
- `evaluation_images.csv`：实际评估图片列表。
- `summary_conditions.csv`、`summary_overall.csv`：按自然／P 人工／D 人工分别汇总。
- `checkpoints/`：传入 `--save-checkpoints` 时保存最终权重。

本地回归测试（在 code 目录）：`python -m unittest discover -s tests -v`。
