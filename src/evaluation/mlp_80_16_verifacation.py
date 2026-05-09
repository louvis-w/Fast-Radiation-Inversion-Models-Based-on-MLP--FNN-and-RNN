import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
warnings.filterwarnings('ignore')

# ===================== 修复：添加NumPy安全全局变量（解决反序列化报错） =====================
try:
    from torch.serialization import add_safe_globals
    import numpy.core.multiarray
    # 注册NumPy重构函数到安全列表
    add_safe_globals([numpy.core.multiarray._reconstruct])
except:
    pass

# ===================== 1. 配置参数（匹配训练脚本） =====================
class Config:
    # 数据路径
    ml_data_path = "E:/ERA5-ml-0.25-all-20100201-0000.nc"
    rad_data_path = "E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"
    
    # 80层+16参数配置
    n_levels = 80
    input_dim = 324
    task_output_dim = n_levels
    hidden_dims = [1024, 512, 256]
    dropout_rate = 0.1
    batch_norm_eps = 1e-5
    task_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    
    # 可视化配置
    sample_num = 80000  # 匹配训练样本数
    fig_size = (15, 12)  # 单参数图表尺寸
    dpi = 300  # 高分辨率输出
    # 可视化结果存储文件夹（自动创建）
    save_root_dir = "16params_visualization_results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ===================== 2. 创建存储文件夹 =====================
def create_dirs():
    """创建主文件夹+每个参数的子文件夹"""
    if not os.path.exists(config.save_root_dir):
        os.makedirs(config.save_root_dir)
    # 为每个参数创建子文件夹
    for var in config.task_vars:
        var_dir = os.path.join(config.save_root_dir, var)
        if not os.path.exists(var_dir):
            os.makedirs(var_dir)
    print(f"可视化结果将存储至: {config.save_root_dir}")

# ===================== 3. 单任务数据集（与训练脚本一致） =====================
class ERA5SingleTaskDataset(Dataset):
    def __init__(self, ml_data, rad_data, task_var, normalize=True):
        self.normalize = normalize
        self.task_var = task_var
        self.var_idx = config.task_vars.index(task_var)
        self.n_levels = config.n_levels
        
        # 裁剪为底部80层数据
        ml_data = ml_data[:, :config.n_levels * len(config.task_vars)]
        self.task_ml_data = ml_data[:, self.var_idx::len(config.task_vars)]
        
        # 数据预处理
        rad_data = np.clip(rad_data, -1e4, 1e4)
        self.task_ml_data = np.clip(self.task_ml_data, -1e4, 1e4)
        
        # NaN填充
        rad_data = np.nan_to_num(rad_data, nan=0.0, posinf=1e4, neginf=-1e4)
        self.task_ml_data = np.nan_to_num(self.task_ml_data, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 单独归一化
        if normalize:
            self.rad_min = np.min(rad_data, axis=0)
            self.rad_max = np.max(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            self.ml_min = np.min(self.task_ml_data, axis=0)
            self.ml_max = np.max(self.task_ml_data, axis=0)
            equal_mask = self.ml_max == self.ml_min
            self.ml_max[equal_mask] = self.ml_min[equal_mask] + 1e-8
            self.task_ml_data = 2 * (self.task_ml_data - self.ml_min) / (self.ml_max - self.ml_min) - 1
        
        # 最终裁剪
        self.task_ml_data = np.clip(self.task_ml_data, -1, 1)
        rad_data = np.clip(rad_data, -1, 1)
        
        self.rad_data = torch.FloatTensor(rad_data)
        self.task_ml_data = torch.FloatTensor(self.task_ml_data)

    def __len__(self):
        return len(self.rad_data)

    def __getitem__(self, idx):
        return self.rad_data[idx], self.task_ml_data[idx]
    
    def inverse_normalize(self, normalized_data):
        """反归一化当前参数"""
        if not self.normalize:
            return normalized_data
        if len(normalized_data.shape) == 1:
            normalized_data = normalized_data.reshape(1, -1)
        return (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min

# ===================== 4. 80层数据加载函数 =====================
def load_80level_data():
    ml_ds = xr.open_dataset(config.ml_data_path)
    rad_ds = xr.open_dataset(config.rad_data_path)
    
    # 加载亮温数据
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    rad_valid = ~np.all(np.isnan(rad_data), axis=1)
    rad_data = rad_data[rad_valid]
    
    # 加载80层大气参数
    ml_vars = config.task_vars
    ml_data_list = []
    for level in range(config.n_levels):
        level_data = []
        for var in ml_vars:
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 内存安全裁剪
    if len(ml_data) > config.sample_num:
        idx = np.random.choice(len(ml_data), config.sample_num, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    print(f"80层数据加载完成 - 总样本数: {len(ml_data)}")
    return ml_data, rad_data

# ===================== 5. 单任务MLP模型（与训练脚本一致） =====================
class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(SingleTaskMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim, eps=bn_eps),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate) if i < len(hidden_dims)-1 else nn.Identity()
            ])
            prev_dim = dim
        
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ===================== 6. 单参数可视化核心函数 =====================
def visualize_single_param(task_var, ml_data, rad_data):
    """生成单个参数的全套可视化图表"""
    # 1. 加载单参数数据集和模型
    var_dir = os.path.join(config.save_root_dir, task_var)
    model_path = f"best_mlp_80level_{task_var}.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告：未找到{task_var}的模型文件 {model_path}，跳过该参数")
        return
    
    # 创建数据集
    dataset = ERA5SingleTaskDataset(ml_data, rad_data, task_var)
    # 取20%验证集
    total_samples = len(dataset)
    val_size = int(0.2 * total_samples)
    train_size = total_samples - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 加载模型（修复核心：设置weights_only=False并捕获异常）
    model = SingleTaskMLP(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        hidden_dims=config.hidden_dims
    ).to(config.device)
    
    try:
        # 修复：添加weights_only=False解决反序列化报错
        checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"错误：加载{task_var}模型失败 - {str(e)}")
        return
    
    # 2. 批量预测
    all_pred = []
    all_true = []
    with torch.no_grad():
        for rad_batch, ml_batch in val_loader:
            rad_batch = rad_batch.to(config.device)
            ml_batch = ml_batch.to(config.device)
            
            pred_batch = model(rad_batch)
            # 反归一化
            pred_np = dataset.inverse_normalize(pred_batch.cpu().numpy())
            true_np = dataset.inverse_normalize(ml_batch.cpu().numpy())
            
            all_pred.append(pred_np)
            all_true.append(true_np)
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    # 3. 计算核心评估指标
    mae = np.mean(np.abs(all_pred - all_true))
    mse = np.mean((all_pred - all_true) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 保存指标到txt文件
    metrics_path = os.path.join(var_dir, f"{task_var}_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"========== {task_var} (80层) 模型性能指标 ==========\n")
        f.write(f"平均绝对误差 (MAE): {mae:.6f}\n")
        f.write(f"均方误差 (MSE): {mse:.6f}\n")
        f.write(f"均方根误差 (RMSE): {rmse:.6f}\n")
        f.write(f"决定系数 (R²): {r2:.6f}\n")
    print(f"{task_var} 指标已保存至: {metrics_path}")
    
    # 4. 生成可视化图表（4个子图）
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=config.fig_size, dpi=config.dpi)
    fig.suptitle(f"{task_var} - 80层参数反演效果 (MAE={mae:.4f}, R²={r2:.4f})", fontsize=14, y=0.98)
    
    # 子图1：整体误差分布直方图
    ax1 = axes[0, 0]
    total_error = all_pred - all_true
    ax1.hist(total_error.flatten(), bins=80, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax1.set_xlabel('预测误差')
    ax1.set_ylabel('频次')
    ax1.set_title('整体误差分布')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 子图2：第0层参数真实vs预测散点图
    ax2 = axes[0, 1]
    level_0_true = all_true[:, 0]
    level_0_pred = all_pred[:, 0]
    # 随机采样1000个点避免重叠
    sample_idx = np.random.choice(len(level_0_true), min(1000, len(level_0_true)), replace=False)
    ax2.scatter(level_0_true[sample_idx], level_0_pred[sample_idx], 
                alpha=0.6, s=10, color='coral', edgecolor='none')
    # 添加y=x参考线
    min_val = min(level_0_true.min(), level_0_pred.min())
    max_val = max(level_0_true.max(), level_0_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel(f'{task_var} 真实值 (第0层)')
    ax2.set_ylabel(f'{task_var} 预测值 (第0层)')
    ax2.set_title('第0层参数：真实值 vs 预测值')
    ax2.grid(alpha=0.3)
    
    # 子图3：80层逐层MAE曲线
    ax3 = axes[1, 0]
    level_mae = []
    for level in range(config.n_levels):
        level_error = np.mean(np.abs(all_pred[:, level] - all_true[:, level]))
        level_mae.append(level_error)
    ax3.plot(range(config.n_levels), level_mae, color='darkgreen', linewidth=2, marker='.', markersize=4)
    ax3.set_xlabel('大气层级 (0-79层)')
    ax3.set_ylabel('平均绝对误差 (MAE)')
    ax3.set_title('80层逐层MAE变化')
    ax3.grid(alpha=0.3)
    
    # 子图4：参数分布对比（核密度图）
    ax4 = axes[1, 1]
    # 取第0层数据做分布对比
    sns.kdeplot(level_0_true, label='真实值', ax=ax4, color='blue', linewidth=2)
    sns.kdeplot(level_0_pred, label='预测值', ax=ax4, color='orange', linewidth=2)
    ax4.set_xlabel(f'{task_var} 数值 (第0层)')
    ax4.set_ylabel('概率密度')
    ax4.set_title('参数分布对比')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    fig_path = os.path.join(var_dir, f"{task_var}_performance.png")
    plt.savefig(fig_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"{task_var} 可视化图表已保存至: {fig_path}")

# ===================== 7. 批量可视化16个参数 =====================
def batch_visualize():
    # 1. 创建文件夹
    create_dirs()
    
    # 2. 加载80层数据
    print("加载80层验证数据...")
    ml_data, rad_data = load_80level_data()
    
    # 3. 逐个参数可视化
    print("\n开始批量生成16个参数的可视化结果...")
    for idx, var in enumerate(config.task_vars):
        print(f"\n[{idx+1}/16] 处理参数: {var}")
        visualize_single_param(var, ml_data, rad_data)
    
    print(f"\n所有参数可视化完成！结果存储至: {config.save_root_dir}")

# ===================== 8. 主执行流程 =====================
if __name__ == "__main__":
    batch_visualize()