# Machado 2009 色觉缺陷模拟复现

本文件夹复现了论文中的红-绿色觉缺陷模拟方法：

Machado, Oliveira, and Fernandes, "A Physiologically-based Model for Simulation of Color Vision Deficiency", IEEE TVCG, 2009.

## 文件说明

- `machado_matrices.json`：保存 Machado 2009 模型常用的 3x3 线性 RGB 模拟矩阵，包含 `protanomaly` 和 `deuteranomaly`，严重程度使用归一化数值 `0.0` 到 `1.0`。
- `simulate_cvd.py`：命令行图片模拟脚本。脚本会先把普通 sRGB 图片转换到线性 sRGB，应用矩阵后再转换回普通 sRGB。

矩阵严重程度说明：

- `0.0` 表示正常三色视觉。
- `1.0` 表示对应类型的二色视觉端点。
- `protanomaly` 的 `1.0` 对应 `protanopia`，即红色盲。
- `deuteranomaly` 的 `1.0` 对应 `deuteranopia`，即绿色盲。

注意：论文正文中描述红色弱、绿色弱程度时主要使用光谱峰值偏移量，单位是 nm，例如 2 nm、8 nm、14 nm、20 nm，并没有把正文实验固定成 `0.0` 到 `1.0` 的 10 个等级。这里的 `0.0` 到 `1.0` 是矩阵实现中常用的归一化 severity 表示；可以近似理解为从正常视觉连续过渡到对应二色视觉端点。

## 使用方法

```bash
python3 simulate_cvd.py input.png output_protanopia.png --type protanopia --severity 1.0
python3 simulate_cvd.py input.png output_protan_weak.png --type protanomaly --severity 0.5
python3 simulate_cvd.py input.png output_deutanopia.png --type deuteranopia --severity 1.0
python3 simulate_cvd.py input.png output_deutan_weak.png --type deuteranomaly --severity 0.5
```

参数说明：

- `input.png`：输入图片路径。
- `output_*.png`：输出图片路径。
- `--type`：色觉缺陷类型。
- `--severity`：严重程度，取值范围为 `[0.0, 1.0]`。

`--severity` 可以使用任意 `0.0` 到 `1.0` 之间的小数。如果不是矩阵表中已有的 0.1 间隔值，脚本会在相邻两个矩阵之间做线性插值。

## 支持的类型别名

脚本支持以下 `--type` 写法：

- 红色觉相关：`red`、`protan`、`protanomaly`、`protanopia`
- 绿色觉相关：`green`、`deutan`、`deuteranomaly`、`deuteranopia`

## 示例

模拟红色盲：

```bash
python3 simulate_cvd.py test.png test_protanopia.png --type protanopia --severity 1.0
```

模拟中等程度红色弱：

```bash
python3 simulate_cvd.py test.png test_protanomaly_05.png --type protanomaly --severity 0.5
```

模拟绿色盲：

```bash
python3 simulate_cvd.py test.png test_deuteranopia.png --type deuteranopia --severity 1.0
```

模拟较轻程度绿色弱：

```bash
python3 simulate_cvd.py test.png test_deuteranomaly_03.png --type deuteranomaly --severity 0.3
```
