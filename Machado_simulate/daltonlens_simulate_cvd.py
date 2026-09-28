from PIL import Image
import numpy as np
from daltonlens import simulate

“”“
daltonlens 是一个用于模拟色盲和色弱的库，它使用 Machado 2009 矩阵来模拟色盲和色弱。
它支持以下几种色盲和色弱：
- 绿色盲/绿色弱 (Deutan)
- 红色盲/红色弱 (Protan)
- 蓝色盲/蓝色弱 (Tritan)
- 绿色盲/绿色弱 (Deutan)
- 红色盲/红色弱 (Protan)
- 蓝色盲/蓝色弱 (Tritan)
”“”

# 1. 读取你想模拟的图片，并转换为 NumPy 数组 (RGB 格式)
img = Image.open("data/ sample.png").convert("RGB")
img_array = np.asarray(img)

# 2. 初始化 Machado 2009 模拟器
simulator = simulate.Simulator_Machado2009()

# 3. 模拟绿色盲/绿色弱 (Deutan)
# severity=1.0 表示完全的绿色盲 (Deuteranopia)
# 如果想要模拟轻度或中度色弱，可以将 severity 设为 0.5 等中间值
simulated_deutan = simulator.simulate_cvd(
    img_array, 
    deficiency=simulate.Deficiency.DEUTAN, 
    severity=0.5
)

# 4. 模拟红色盲/红色弱 (Protan)
simulated_protan = simulator.simulate_cvd(
    img_array,
    deficiency=simulate.Deficiency.PROTAN, 
    severity=0.5
)

# 5. 将处理后的数组重新转回 PIL Image 并保存
Image.fromarray(simulated_deutan).save("results_dalton/deutan_simulated.png")
Image.fromarray(simulated_protan).save("results_dalton/protan_simulated.png")

print("模拟图片已成功生成！")