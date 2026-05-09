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

# ===================== 1. 配置参数（补全缺失的val_ratio/train_ratio） =====================
class Config:
    # 路径配置（修改为你的模型保存路径）
    ml_data_path = "E:/ERA5-ml-0.25-all-20100201-0000.nc"
    rad_data_path = "E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"
    model_path = "best_mlp_137level_large.pth"  # 你的模型权重路径
    
    # 模型参数（与训练脚本完全一致）
    n_levels = 137
    input_dim = 324
    output_dim = n_levels * 16
    hidden_dims = [4096, 2048, 1024, 512]  # 关键：匹配训练时的隐藏层维度
    dropout_rate = 0.1
    batch_norm_eps = 1e-5
    
    # 训练划分参数（补全缺失的val_ratio/train_ratio）
    train_ratio = 0.8
    val_ratio = 0.2
    
    # 可视化配置
    sample_num = 80000  # 匹配训练的样本数
    plot_cols = 4
    fig_size = (20, 15)
    save_fig_path = "mlp_137level_performance.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 大气参数列表（与训练一致）
    ml_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]

config = Config()

# ===================== 2. 数据集定义（与训练脚本完全一致） =====================
class ERA5Dataset(Dataset):
    def __init__(self, ml_data, rad_data, normalize=True):
        self.normalize = normalize
        self.ml_vars = config.ml_vars
        self.n_levels = config.n_levels
        
        # 第一步：全局数值裁剪
        ml_data = np.clip(ml_data, -1e4, 1e4)
        rad_data = np.clip(rad_data, -1e4, 1e4)
        
        # 第二步：鲁棒最大最小归一化
        if normalize:
            # 亮温归一化
            self.rad_min = np.nanmin(rad_data, axis=0)
            self.rad_max = np.nanmax(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            # 大气参数归一化
            self.ml_min = np.nanmin(ml_data, axis=0)
            self.ml_max = np.nanmax(ml_data, axis=0)
            equal_mask = self.ml_max == self.ml_min
            self.ml_max[equal_mask] = self.ml_min[equal_mask] + 1e-8
            ml_data = 2 * (ml_data - self.ml_min) / (self.ml_max - self.ml_min) - 1
        
        # 第三步：最终裁剪+NaN填充
        ml_data = np.clip(ml_data, -1, 1)
        rad_data = np.clip(rad_data, -1, 1)
        ml_data = np.nan_to_num(ml_data, nan=0.0, posinf=1.0, neginf=-1.0)
        rad_data = np.nan_to_num(rad_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.rad_data = torch.FloatTensor(rad_data)
        self.ml_data = torch.FloatTensor(ml_data)

    def __len__(self):
        return len(self.rad_data)

    def __getitem__(self, idx):
        return self.rad_data[idx], self.ml_data[idx]
    
    def inverse_normalize_ml(self, normalized_data):
        """反归一化大气参数到原始尺度"""
        if not self.normalize:
            return normalized_data
        # 处理批量数据和单样本
        if len(normalized_data.shape) == 1:
            normalized_data = normalized_data.reshape(1, -1)
        return (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min

# ===================== 3. 数据加载函数（与训练脚本一致） =====================
def load_137level_data(ml_path, rad_path):
    ml_ds = xr.open_dataset(ml_path)
    rad_ds = xr.open_dataset(rad_path)
    
    ml_vars = config.ml_vars
    
    # 加载亮温数据
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    rad_valid = ~np.all(np.isnan(rad_data), axis=1)
    rad_data = rad_data[rad_valid]
    
    # 加载137层大气参数
    ml_data_list = []
    max_samples = config.sample_num
    for level in range(config.n_levels):
        level_data = []
        for var in ml_vars:
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 内存安全裁剪（匹配训练）
    if len(ml_data) > max_samples:
        idx = np.random.choice(len(ml_data), max_samples, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    print(f"验证数据加载完成 - 总样本数: {len(ml_data)}")
    print(f"大气参数维度: {ml_data.shape}, 亮温维度: {rad_data.shape}")
    return ml_data, rad_data

# ===================== 4. 模型定义（与训练脚本完全一致） =====================
class LargeMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(LargeMLP, self).__init__()
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

# ===================== 5. 核心可视化函数 =====================
def visualize_model():
    # 1. 加载验证数据
    print("加载验证数据...")
    ml_data, rad_data = load_137level_data(config.ml_data_path, config.rad_data_path)
    dataset = ERA5Dataset(ml_data, rad_data)
    
    # 划分验证集（取20%，匹配训练）
    total_samples = len(dataset)
    val_size = int(config.val_ratio * total_samples)
    train_size = total_samples - val_size
    torch.manual_seed(42)
    np.random.seed(42)
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 2. 加载训练好的模型
    print("加载模型权重...")
    model = LargeMLP(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
        bn_eps=config.batch_norm_eps
    ).to(config.device)
    
    # 加载权重（忽略无关警告）
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()
    
    # 3. 批量预测
    print("批量预测验证样本...")
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for rad_batch, ml_batch in val_loader:
            rad_batch = rad_batch.to(config.device)
            ml_batch = ml_batch.to(config.device)
            
            pred_batch = model(rad_batch)
            
            # 反归一化
            pred_np = dataset.inverse_normalize_ml(pred_batch.cpu().numpy())
            true_np = dataset.inverse_normalize_ml(ml_batch.cpu().numpy())
            
            all_pred.append(pred_np)
            all_true.append(true_np)
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    
    # 4. 可视化配置
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = config.fig_size
    fig = plt.figure(figsize=config.fig_size)
    gs = fig.add_gridspec(4, config.plot_cols, hspace=0.4, wspace=0.3)
    
    # ========== 子图1：整体误差分布 ==========
    ax1 = fig.add_subplot(gs[0, 0:2])
    total_error = all_pred - all_true
    ax1.hist(total_error.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Error Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # ========== 子图2：温度(T)真实vs预测散点图 ==========
    ax2 = fig.add_subplot(gs[0, 2:4])
    t_idx = config.ml_vars.index('t')  # 温度参数索引
    level_0_t_true = all_true[:, t_idx]
    level_0_t_pred = all_pred[:, t_idx]
    
    # 随机采样1000个点避免重叠
    sample_idx = np.random.choice(len(level_0_t_true), min(1000, len(level_0_t_true)), replace=False)
    ax2.scatter(level_0_t_true[sample_idx], level_0_t_pred[sample_idx], 
                alpha=0.6, s=8, color='coral', edgecolor='none')
    # 添加y=x参考线
    min_val = min(level_0_t_true.min(), level_0_t_pred.min())
    max_val = max(level_0_t_true.max(), level_0_t_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('True T (Level 0)')
    ax2.set_ylabel('Pred T (Level 0)')
    ax2.set_title('Temperature (t) - True vs Predicted')
    ax2.grid(alpha=0.3)
    
    # ========== 子图3：137层逐层平均误差 ==========
    ax3 = fig.add_subplot(gs[1, :])
    level_errors = []
    for level in range(config.n_levels):
        start = level * len(config.ml_vars)
        end = (level + 1) * len(config.ml_vars)
        level_error = np.mean(np.abs(all_pred[:, start:end] - all_true[:, start:end]))
        level_errors.append(level_error)
    
    ax3.plot(range(config.n_levels), level_errors, color='darkgreen', linewidth=2, marker='.', markersize=4)
    ax3.set_xlabel('Atmospheric Level (0-136)')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('MAE by Atmospheric Level (137 Levels)')
    ax3.grid(alpha=0.3)
    
    # ========== 子图4：各参数平均误差对比 ==========
    ax4 = fig.add_subplot(gs[2, :])
    var_errors = []
    for var_idx, var_name in enumerate(config.ml_vars):
        var_pos = [var_idx + level * len(config.ml_vars) for level in range(config.n_levels)]
        var_error = np.mean(np.abs(all_pred[:, var_pos] - all_true[:, var_pos]))
        var_errors.append(var_error)
    
    bars = ax4.bar(config.ml_vars, var_errors, color='lightseagreen', alpha=0.8)
    ax4.set_xlabel('Atmospheric Parameters')
    ax4.set_ylabel('Mean Absolute Error (All Levels)')
    ax4.set_title('MAE by Parameter')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3, axis='y')
    
    # 标注数值
    for bar, err in zip(bars, var_errors):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{err:.3f}', ha='center', va='bottom', fontsize=8)
    
    # ========== 子图5：湿度(Q)逐层误差 ==========
    ax5 = fig.add_subplot(gs[3, 0:2])
    q_idx = config.ml_vars.index('q')
    q_errors = []
    for level in range(config.n_levels):
        q_pos = q_idx + level * len(config.ml_vars)
        q_err = np.mean(np.abs(all_pred[:, q_pos] - all_true[:, q_pos]))
        q_errors.append(q_err)
    
    ax5.plot(range(config.n_levels), q_errors, color='purple', linewidth=2, marker='.', markersize=4)
    ax5.set_xlabel('Atmospheric Level (0-136)')
    ax5.set_ylabel('MAE')
    ax5.set_title('Humidity (q) Error by Level')
    ax5.grid(alpha=0.3)
    
    # ========== 子图6：风速(U)分布对比 ==========
    ax6 = fig.add_subplot(gs[3, 2:4])
    u_idx = config.ml_vars.index('u')
    level_0_u_true = all_true[:, u_idx]
    level_0_u_pred = all_pred[:, u_idx]
    
    sns.kdeplot(level_0_u_true, label='True U', ax=ax6, color='blue', linewidth=2)
    sns.kdeplot(level_0_u_pred, label='Pred U', ax=ax6, color='orange', linewidth=2)
    ax6.set_xlabel('U Wind Speed (Level 0)')
    ax6.set_ylabel('Density')
    ax6.set_title('Wind Speed (u) - True vs Pred Distribution')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 5. 保存和显示
    plt.suptitle('137-Level MLP Model Performance Visualization', fontsize=16, y=0.98)
    plt.savefig(config.save_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存至: {config.save_fig_path}")
    plt.show()
    
    # 6. 打印核心评估指标
    mae = np.mean(np.abs(all_pred - all_true))
    mse = np.mean((all_pred - all_true) ** 2)
    rmse = np.sqrt(mse)
    # 计算R²（避免除0）
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print("\n========== 模型性能指标 ==========")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print("==================================")

# ===================== 6. 主执行 =====================
if __name__ == "__main__":
    # 检查模型文件
    if not os.path.exists(config.model_path):
        print(f"错误：未找到模型文件 {config.model_path}")
        print("请确认模型路径是否正确，或先运行训练脚本生成模型")
    else:
        visualize_model()