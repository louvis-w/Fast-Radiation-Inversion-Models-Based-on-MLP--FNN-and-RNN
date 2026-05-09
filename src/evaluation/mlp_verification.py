import os
import numpy as np
import xarray as xr
import torch
# 补充缺失的nn导入！！
import torch.nn as nn  
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 配置参数 =====================
class Config:
    # 路径配置（请根据实际路径修改）
    ml_data_path = "F:/ERA5-ml-0.25-all-20100201-1200.nc"          
    rad_data_path = "F:/ERA5-rad-0.25-20100201-1200.nocloud.airs324_aqua.crtm.nc"  
    model_path = "best_mlp_model.pth"  # 训练好的模型权重文件
    save_fig_dir = "model_evaluation_plots"  # 可视化结果保存目录
    
    # 模型参数（必须与训练时完全一致！）
    input_dim = 324
    output_dim = 16
    hidden_dims = [1024, 512, 256]
    dropout_rate = 0.2
    
    # 大气参数名称（用于可视化标注）
    ml_var_names = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# 创建可视化目录
os.makedirs(config.save_fig_dir, exist_ok=True)

# ===================== 2. 数据集类（与训练代码一致） =====================
class ERA5Dataset(Dataset):
    def __init__(self, ml_data, rad_data, normalize=True):
        self.normalize = normalize
        
        if normalize:
            # 亮温数据归一化
            self.rad_mean = np.mean(rad_data, axis=0)
            self.rad_std = np.std(rad_data, axis=0)
            rad_data = (rad_data - self.rad_mean) / (self.rad_std + 1e-8)
            
            # 大气参数归一化
            self.ml_mean = np.mean(ml_data, axis=0)
            self.ml_std = np.std(ml_data, axis=0)
            ml_data = (ml_data - self.ml_mean) / (self.ml_std + 1e-8)

        self.rad_data = torch.FloatTensor(rad_data)
        self.ml_data = torch.FloatTensor(ml_data)

    def __len__(self):
        return len(self.rad_data)

    def __getitem__(self, idx):
        return self.rad_data[idx], self.ml_data[idx]
    
    def inverse_normalize_ml(self, normalized_data):
        """反归一化大气参数（新增：用于预测结果还原）"""
        return normalized_data * self.ml_std + self.ml_mean

# ===================== 3. 数据加载函数（与训练代码一致） =====================
def load_single_file_data(ml_path, rad_path):
    ml_ds = xr.open_dataset(ml_path)
    rad_ds = xr.open_dataset(rad_path)
    
    ml_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    
    ml_data_list = []
    for var in ml_vars:
        var_data = ml_ds[var].isel(time=0, level=0).values
        var_data = np.nan_to_num(var_data, nan=np.nanmean(var_data))
        ml_data_list.append(var_data.flatten())
    
    ml_data = np.stack(ml_data_list, axis=1)
    
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    
    valid_mask = ~(np.isnan(ml_data).any(axis=1) | np.isnan(rad_data).any(axis=1))
    ml_data = ml_data[valid_mask]
    rad_data = rad_data[valid_mask]
    
    return ml_data, rad_data

# ===================== 4. MLP模型定义（与训练代码完全一致） =====================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ===================== 5. 可视化函数 =====================
def plot_pred_vs_true(true_data, pred_data, var_names, save_dir):
    """绘制前5个参数的预测值vs真实值散点图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(5):  # 绘制前5个参数
        ax = axes[i]
        # 随机采样10000个点（避免点过多）
        sample_idx = np.random.choice(len(true_data), min(10000, len(true_data)), replace=False)
        true_sample = true_data[sample_idx, i]
        pred_sample = pred_data[sample_idx, i]
        
        ax.scatter(true_sample, pred_sample, alpha=0.5, s=1, c='#1f77b4')
        # 绘制y=x参考线
        min_val = min(np.min(true_sample), np.min(pred_sample))
        max_val = max(np.max(true_sample), np.max(pred_sample))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax.set_xlabel(f'True {var_names[i]}', fontsize=10)
        ax.set_ylabel(f'Pred {var_names[i]}', fontsize=10)
        ax.set_title(f'{var_names[i]}: Pred vs True', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 隐藏多余的子图
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pred_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rmse(rmse_list, var_names, save_dir):
    """绘制各参数RMSE柱状图"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(var_names)))
    bars = plt.bar(var_names, rmse_list, color=colors)
    
    # 标注RMSE值
    for bar, rmse in zip(bars, rmse_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{rmse:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Atmospheric Parameters', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE of Each Atmospheric Parameter', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_param_rmse.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ===================== 6. 主验证流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    print("Loading data...")
    ml_data, rad_data = load_single_file_data(config.ml_data_path, config.rad_data_path)
    print(f"Data loaded: ML shape={ml_data.shape}, Rad shape={rad_data.shape}")
    
    # 2. 创建数据集（复用训练时的归一化逻辑）
    dataset = ERA5Dataset(ml_data, rad_data)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    # 3. 加载模型
    print(f"Loading model from {config.model_path}...")
    model = MLP(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()  # 切换到评估模式（关闭Dropout/BatchNorm）
    
    # 4. 批量预测
    print("Generating predictions...")
    all_preds = []
    all_trues = []
    
    with torch.no_grad():  # 关闭梯度计算，提升速度
        for rad_batch, ml_batch in dataloader:
            rad_batch = rad_batch.to(config.device)
            pred_batch = model(rad_batch).cpu().numpy()
            true_batch = ml_batch.numpy()
            
            # 反归一化还原真实值
            pred_batch = dataset.inverse_normalize_ml(pred_batch)
            true_batch = dataset.inverse_normalize_ml(true_batch)
            
            all_preds.append(pred_batch)
            all_trues.append(true_batch)
    
    # 合并结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # 5. 计算评估指标
    print("Calculating evaluation metrics...")
    # 整体RMSE
    overall_rmse = np.sqrt(np.mean((all_preds - all_trues) **2))
    # 各参数RMSE
    per_param_rmse = [
        np.sqrt(np.mean((all_preds[:, i] - all_trues[:, i])**2)) 
        for i in range(config.output_dim)
    ]
    
    # 打印结果
    print(f"\n===== Evaluation Results =====")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print("\nPer Parameter RMSE:")
    for name, rmse in zip(config.ml_var_names, per_param_rmse):
        print(f"{name}: {rmse:.6f}")
    
    # 6. 生成可视化图表
    print("\nGenerating plots...")
    plot_pred_vs_true(all_trues, all_preds, config.ml_var_names, config.save_fig_dir)
    plot_rmse(per_param_rmse, config.ml_var_names, config.save_fig_dir)
    
    print(f"\nAll results saved to {config.save_fig_dir}")
    print("Model verification completed!")