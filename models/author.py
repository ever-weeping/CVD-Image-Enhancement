import sys
from pathlib import Path

# 将项目根目录下的 vendor 文件夹动态添加到 Python 模块搜索路径的最前面
# 这样可以优先导入 vendor 目录下的自定义模块，避免与系统模块冲突
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor"))

# discrimintor_trs 是一个基于 Transformer 架构的深度学习模型模块
# 其中的 Generator_transformer_pathch4_844_48_3_nouplayer_server5 是一个生成器网络类
# 类名含义解析：
#   - Generator_transformer: 基于 Transformer 的生成器模型
#   - pathch4: 使用 patch 大小为 4 的图像分块策略
#   - 844_48_3: 网络结构超参数（可能包含通道数、层数、注意力头数等配置）
#   - nouplayer: 无上采样层（no uplayer），表示网络中不包含上采样操作
#   - server5: 服务器版本号或配置编号，用于区分不同的实验配置
from discrimintor_trs import Generator_transformer_pathch4_844_48_3_nouplayer_server5

def make_model():
    """
    模型工厂函数：创建并返回一个生成器模型实例。
    
    Returns:
        Generator_transformer_pathch4_844_48_3_nouplayer_server5: 
            一个基于 Transformer 的生成器模型实例，用于后续的模型训练或推理。
    """
    return Generator_transformer_pathch4_844_48_3_nouplayer_server5()
