# Deep Learning Radiative Inversion (深度学习快速辐射反演模型)

基于深度学习的卫星辐射数据快速大气参数反演系统，使用 IASI/AIRS 324通道亮温数据反演16个大气参数的垂直廓线（80层）。

## 项目简介

本项目实现了多种深度学习神经网络架构用于卫星辐射数据的快速大气反演：

- **MLP (多层感知机)**: 基础的全连接网络架构
- **Physics-FNN (物理约束前馈网络)**: 融合大气物理先验知识的神经网络
- **支持多层级反演**: 80层/137层垂直分辨率

## 反演参数（16个大气参数）

| 参数 | 英文 | 单位 |
|------|------|------|
| 温度 | t | K |
| 比湿 | q | kg/kg |
| 纬向风 | u | m/s |
| 经向风 | v | m/s |
| 垂直速度 | w | Pa/s |
| 位势高度 | z | m |
| 气压对数 | lnsp | - |
| 臭氧 | o3 | kg/kg |
| 云量 | cc | - |
| 云冰水含量 | ciwc | kg/kg |
| 云液水含量 | clwc | kg/kg |
| 雨水含量 | crwc | kg/kg |
| 雪水含量 | cswc | kg/kg |
| 散射率 | d | - |
| 涡度 | vo | s⁻¹ |
| eta点导数 | etadot | - |

## 项目结构

```
Deep-Learning-Radiative-Inversion/
├── README.md                    # 项目说明
├── .gitignore                   # Git忽略规则
├── requirements.txt             # Python依赖
├── src/
│   ├── models/                  # 模型定义
│   │   ├── fnn.py              # 物理约束FNN模型
│   │   ├── mlp.py              # 基础MLP模型(16参数)
│   │   ├── mlp_80.py           # MLP模型(80层+16任务)
│   │   └── ...
│   ├── training/                # 训练相关脚本
│   └── evaluation/              # 验证与评估脚本
├── pretrained_models/           # 预训练模型权重
│   ├── mlp_80level/            # MLP 80层模型
│   ├── mlp_midlow80level/      # MLP 中低层80层模型
│   ├── fnn_80level/            # Physics-FNN 80层模型
│   ├── fnn_midlow80level/      # FNN 中低层80层模型
│   └── other/                  # 其他模型(137层等)
└── results/                     # 实验结果
    ├── global_analysis/        # 全球分析图
    ├── evaluation_plots/       # 模型评估图
    ├── param_visualization/    # 各参数可视化结果
    ├── best_level_analysis/    # 最佳层级分析
    └── visualizations/         # 其他可视化结果
```

## 环境要求

### Python依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy
- xarray
- tqdm
- matplotlib (用于可视化)

### 硬件建议

- **GPU**: NVIDIA GPU (推荐显存 >= 8GB)
- **内存**: >= 16GB RAM

## 快速开始

### 1. 数据准备

项目使用 ERA5 再分析数据和 IASI/AIRS 辐射数据：
- 大气参数数据: ERA5 ML (气象场)
- 辐射观测数据: IASI/AIRS L1B 亮温 (324通道)

### 2. 训练模型

```bash
# 训练基础MLP模型 (16参数输出)
python src/models/mlp.py

# 训练80层MLP模型 (每个参数独立训练)
python src/models/mlp_80.py

# 训练物理约束FNN模型
python src/models/fnn.py
```

### 3. 模型验证

```bash
# 验证MLP模型
python src/evaluation/mlp_verification.py

# 验证FNN模型
python src/evaluation/fnn_verification.py
```

### 4. 使用预训练模型推理

```python
import torch
from src.models.mlp import MLPModel

# 加载预训练模型
model = MLPModel()
model.load_state_dict(torch.load('pretrained_models/mlp_80level/best_mlp_80level_t.pth'))
model.eval()

# 推理 (输入324通道亮温)
with torch.no_grad():
    output = model(input_radiance)
```

## 模型性能

各参数在测试集上的 RMSE 表现请参见 `results/evaluation_plots/` 目录。

## 主要特性

1. **多模型架构**: 支持MLP、FNN、RNN等多种网络结构
2. **物理约束**: FNN模型融入大气物理先验知识
3. **多层级支持**: 支持80层和137层垂直分辨率
4. **独立任务学习**: 16个参数可独立训练优化
5. **中低层优化**: 针对大气中低层的专门优化版本

## 引用

如果您使用了本项目的代码或模型，请引用：

```bibtex
@misc{dl_radiative_inversion,
  title={Deep Learning Fast Radiative Inversion Model},
  author={Your Name},
  year={2024},
  note={Based on IASI/AIRS satellite radiance data}
}
```

## 许可证

MIT License

## 联系方式
231840270@smail.nju.edu.cn   louvis-w
如有问题，欢迎提 Issue 或联系。
