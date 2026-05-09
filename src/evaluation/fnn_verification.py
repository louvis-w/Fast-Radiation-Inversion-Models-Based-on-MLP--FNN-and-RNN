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

# ===================== 1. 配置参数 =====================
class Config:
    # 数据路径
    ml_data_path = "E:/ERA5-ml-0.25-all-20100201-0000.nc"
    rad_data_path = "E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"
    
    # 模型路径前缀
    model_prefix = "best_physics_fnn_80level_"
    # 80层配置
    n_levels = 80
    input_dim = 324
    task_output_dim = n_levels
    hidden_dims = [1024, 512, 256]
    dropout_rate = 0.1
    batch_norm_eps = 1e-5
    
    # 16个大气参数
    task_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    # 物理值域约束（与训练脚本一致）
    PHYSICS_BOUNDS = {
        't': (180.0, 320.0), 'q': (0.0, 0.04), 'o3': (1e-8, 1e-5),
        'z': (0.0, 100000.0), 'u': (-100.0, 100.0), 'v': (-100.0, 100.0),
        'lnsp': (0.0, 15.0), 'cc': (0.0, 1.0)
    }
    
    # 检验配置
    sample_num = 20000  # 验证样本数（减少计算量）
    batch_size = 64
    fig_size = (18, 10)
    dpi = 300
    # 结果存储文件夹
    result_dir = "FNN_performance_analysis_best_level"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ===================== 2. 创建存储文件夹 =====================
def create_result_dirs():
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    for var in config.task_vars:
        var_dir = os.path.join(config.result_dir, var)
        if not os.path.exists(var_dir):
            os.makedirs(var_dir)
    print(f"检验结果将存储至: {config.result_dir}")

# ===================== 3. 物理约束数据集（与训练脚本一致） =====================
class ERA5PhysicsDataset(Dataset):
    def __init__(self, ml_data, rad_data, task_var, normalize=True):
        self.normalize = normalize
        self.task_var = task_var
        self.var_idx = config.task_vars.index(task_var)
        self.n_levels = config.n_levels
        
        # 裁剪为80层数据
        ml_data = ml_data[:, :config.n_levels * len(config.task_vars)]
        self.task_ml_data = ml_data[:, self.var_idx::len(config.task_vars)]
        
        # 物理值域裁剪
        if task_var in config.PHYSICS_BOUNDS:
            min_val, max_val = config.PHYSICS_BOUNDS[task_var]
            self.task_ml_data = np.clip(self.task_ml_data, min_val, max_val)
        
        # 数值预处理
        rad_data = np.clip(rad_data, -1e4, 1e4)
        self.task_ml_data = np.clip(self.task_ml_data, -1e4, 1e4)
        rad_data = np.nan_to_num(rad_data, nan=0.0, posinf=1e4, neginf=-1e4)
        self.task_ml_data = np.nan_to_num(self.task_ml_data, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 归一化
        if normalize:
            self.rad_min = np.min(rad_data, axis=0)
            self.rad_max = np.max(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            if task_var in config.PHYSICS_BOUNDS:
                self.ml_min = np.full(self.n_levels, config.PHYSICS_BOUNDS[task_var][0])
                self.ml_max = np.full(self.n_levels, config.PHYSICS_BOUNDS[task_var][1])
            else:
                self.ml_min = np.min(self.task_ml_data, axis=0)
                self.ml_max = np.max(self.task_ml_data, axis=0)
            
            equal_mask = self.ml_max == self.ml_min
            if isinstance(equal_mask, bool):
                equal_mask = np.full(self.n_levels, equal_mask)
            self.ml_max[equal_mask] = self.ml_min[equal_mask] + 1e-8
            self.task_ml_data = 2 * (self.task_ml_data - self.ml_min) / (self.ml_max - self.ml_min) - 1
        
        self.task_ml_data = np.clip(self.task_ml_data, -1, 1)
        rad_data = np.clip(rad_data, -1, 1)
        
        self.rad_data = torch.FloatTensor(rad_data)
        self.task_ml_data = torch.FloatTensor(self.task_ml_data)

    def __len__(self):
        return len(self.rad_data)

    def __getitem__(self, idx):
        return self.rad_data[idx], self.task_ml_data[idx]
    
    def inverse_normalize(self, normalized_data):
        if not self.normalize:
            return normalized_data
        if len(normalized_data.shape) == 1:
            normalized_data = normalized_data.reshape(1, -1)
        denorm_data = (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min
        if self.task_var in config.PHYSICS_BOUNDS:
            min_val, max_val = config.PHYSICS_BOUNDS[self.task_var]
            denorm_data = np.clip(denorm_data, min_val, max_val)
        return denorm_data

# ===================== 4. 物理约束FNN模型 =====================
class PhysicsConstrainedFNN(nn.Module):
    def __init__(self, input_dim, output_dim, task_var, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(PhysicsConstrainedFNN, self).__init__()
        self.task_var = task_var
        self.n_levels = config.n_levels
        self.output_dim = output_dim
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0], eps=bn_eps),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i], eps=bn_eps),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Dropout(dropout_rate) if i < len(hidden_dims)-1 else nn.Identity()
                )
            )
        
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.layer_weights = nn.Parameter(torch.ones(3, output_dim))
        
        if task_var in config.PHYSICS_BOUNDS:
            if task_var == 'cc':
                self.output_activation = nn.Sigmoid()
            elif task_var in ['q', 'o3']:
                self.output_activation = nn.Softplus()
            else:
                self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Tanh()
        
        nn.init.constant_(self.layer_weights, 1.0)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)
        
        lower_weight = self.layer_weights[0].unsqueeze(0)
        mid_weight = self.layer_weights[1].unsqueeze(0)
        high_weight = self.layer_weights[2].unsqueeze(0)
        
        out[:, :30] *= lower_weight[:, :30]
        out[:, 30:60] *= mid_weight[:, 30:60]
        out[:, 60:] *= high_weight[:, 60:]
        
        out = self.output_activation(out)
        return out

# ===================== 5. 加载80层数据 =====================
def load_80level_data():
    ml_ds = xr.open_dataset(config.ml_data_path)
    rad_ds = xr.open_dataset(config.rad_data_path)
    
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    rad_valid = ~np.all(np.isnan(rad_data), axis=1)
    rad_data = rad_data[rad_valid]
    
    ml_vars = config.task_vars
    ml_data_list = []
    for level in range(config.n_levels):
        level_data = []
        for var in ml_vars:
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 裁剪样本数
    if len(ml_data) > config.sample_num:
        idx = np.random.choice(len(ml_data), config.sample_num, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    print(f"验证数据加载完成 - 样本数: {len(ml_data)}")
    return ml_data, rad_data

# ===================== 6. 核心函数：自动选择最优层级 + 效果检验 =====================
def analyze_single_param(task_var, ml_data, rad_data):
    """
    对单个参数进行效果检验：
    1. 计算80层MAE，选择最优层级（MAE最小）
    2. 生成最优层级可视化图表
    3. 输出性能指标
    """
    var_dir = os.path.join(config.result_dir, task_var)
    model_path = f"{config.model_prefix}{task_var}.pth"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"警告：未找到{task_var}模型文件 {model_path}，跳过")
        return
    
    # 1. 加载数据集（仅用验证集）
    dataset = ERA5PhysicsDataset(ml_data, rad_data, task_var)
    val_size = int(0.2 * len(dataset))
    _, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 2. 加载模型
    model = PhysicsConstrainedFNN(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        task_var=task_var,
        hidden_dims=config.hidden_dims
    ).to(config.device)
    
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. 批量预测并收集数据
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
    
    all_pred = np.concatenate(all_pred, axis=0)  # (N, 80)
    all_true = np.concatenate(all_true, axis=0)  # (N, 80)
    
    # 4. 计算80层MAE，选择最优层级
    level_mae = []
    for level in range(config.n_levels):
        mae = np.mean(np.abs(all_pred[:, level] - all_true[:, level]))
        level_mae.append(mae)
    
    # 最优层级（MAE最小）
    best_level = np.argmin(level_mae)
    best_mae = level_mae[best_level]
    # 计算最优层级其他指标
    best_pred = all_pred[:, best_level]
    best_true = all_true[:, best_level]
    mse = np.mean((best_pred - best_true) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((best_true - best_pred) ** 2)
    ss_tot = np.sum((best_true - np.mean(best_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 5. 保存性能指标
    metrics_path = os.path.join(var_dir, f"{task_var}_best_level_metrics.txt")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"========== {task_var} 最优层级性能指标 ==========\n")
        f.write(f"最优层级: {best_level} (MAE={best_mae:.6f})\n")
        f.write(f"最优层级RMSE: {rmse:.6f}\n")
        f.write(f"最优层级R²: {r2:.6f}\n")
        f.write(f"最优层级MSE: {mse:.6f}\n\n")
        f.write("========== 80层MAE分布 ==========\n")
        for level in range(config.n_levels):
            f.write(f"层级{level}: MAE={level_mae[level]:.6f}\n")
    
    print(f"\n{task_var} 最优层级: {best_level} | MAE={best_mae:.6f} | R²={r2:.6f}")
    print(f"指标已保存至: {metrics_path}")
    
    # 6. 生成最优层级可视化图表
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=config.fig_size, dpi=config.dpi)
    fig.suptitle(f"{task_var} - 最优层级{best_level}反演效果 (MAE={best_mae:.4f}, R²={r2:.4f})", 
                 fontsize=14, y=0.98)
    
    # 子图1：最优层级真实vs预测散点图（采样2000个点）
    ax1 = axes[0, 0]
    sample_idx = np.random.choice(len(best_true), min(2000, len(best_true)), replace=False)
    ax1.scatter(best_true[sample_idx], best_pred[sample_idx], 
                alpha=0.6, s=8, color='coral', edgecolor='none')
    # 添加y=x参考线
    min_val = min(best_true.min(), best_pred.min())
    max_val = max(best_true.max(), best_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel(f'{task_var} 真实值')
    ax1.set_ylabel(f'{task_var} 预测值')
    ax1.set_title(f'最优层级{best_level}：真实值 vs 预测值')
    ax1.grid(alpha=0.3)
    
    # 子图2：最优层级误差分布直方图
    ax2 = axes[0, 1]
    best_error = best_pred - best_true
    ax2.hist(best_error, bins=60, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差线')
    ax2.set_xlabel('预测误差')
    ax2.set_ylabel('频次')
    ax2.set_title(f'最优层级{best_level}：误差分布')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 子图3：80层MAE分布曲线
    ax3 = axes[0, 2]
    ax3.plot(range(config.n_levels), level_mae, color='darkgreen', linewidth=2, marker='.', markersize=4)
    ax3.axvline(x=best_level, color='red', linestyle='--', linewidth=2, label=f'最优层级{best_level}')
    ax3.set_xlabel('大气层级 (0-79)')
    ax3.set_ylabel('MAE')
    ax3.set_title('80层MAE分布')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 子图4：最优层级参数分布对比（核密度）
    ax4 = axes[1, 0]
    sns.kdeplot(best_true, label='真实值', ax=ax4, color='blue', linewidth=2)
    sns.kdeplot(best_pred, label='预测值', ax=ax4, color='orange', linewidth=2)
    ax4.set_xlabel(f'{task_var} 数值')
    ax4.set_ylabel('概率密度')
    ax4.set_title(f'最优层级{best_level}：分布对比')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 子图5：最优层级误差累积分布
    ax5 = axes[1, 1]
    sorted_error = np.sort(best_error)
    ecdf = np.arange(1, len(sorted_error)+1) / len(sorted_error)
    ax5.plot(sorted_error, ecdf, color='purple', linewidth=2)
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('预测误差')
    ax5.set_ylabel('累积概率')
    ax5.set_title(f'最优层级{best_level}：误差累积分布')
    ax5.grid(alpha=0.3)
    
    # 子图6：随机样本的80层剖面对比（选第1个样本）
    ax6 = axes[1, 2]
    sample_idx_6 = 0
    profile_true = all_true[sample_idx_6, :]
    profile_pred = all_pred[sample_idx_6, :]
    ax6.plot(profile_true, range(config.n_levels), label='真实值', color='blue', linewidth=2)
    ax6.plot(profile_pred, range(config.n_levels), label='预测值', color='orange', linewidth=2)
    ax6.axhline(y=best_level, color='red', linestyle='--', linewidth=2, label=f'最优层级{best_level}')
    ax6.set_xlabel(f'{task_var} 数值')
    ax6.set_ylabel('大气层级 (0-79)')
    ax6.set_title('单个样本80层剖面对比')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    fig_path = os.path.join(var_dir, f"{task_var}_best_level_{best_level}_performance.png")
    plt.savefig(fig_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"{task_var} 可视化图表已保存至: {fig_path}")
    
    return {
        'var': task_var,
        'best_level': best_level,
        'best_mae': best_mae,
        'best_r2': r2,
        'level_mae': level_mae
    }

# ===================== 7. 批量检验所有参数 =====================
def batch_analyze():
    # 创建文件夹
    create_result_dirs()
    
    # 加载数据
    print("加载验证数据...")
    ml_data, rad_data = load_80level_data()
    
    # 批量分析16个参数
    print("\n开始批量分析所有参数（自动选择最优层级）...")
    all_results = []
    for idx, var in enumerate(config.task_vars):
        print(f"\n[{idx+1}/16] 分析参数: {var}")
        result = analyze_single_param(var, ml_data, rad_data)
        if result:
            all_results.append(result)
    
    # 生成汇总报告
    summary_path = os.path.join(config.result_dir, "all_params_best_level_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("========== 16参数最优层级汇总报告 ==========\n")
        f.write("参数\t最优层级\t最优层级MAE\t最优层级R²\n")
        f.write("-" * 60 + "\n")
        for res in all_results:
            f.write(f"{res['var']}\t{res['best_level']}\t{res['best_mae']:.6f}\t{res['best_r2']:.6f}\n")
    
    print(f"\n所有参数分析完成！")
    print(f"汇总报告保存至: {summary_path}")
    print(f"详细结果存储至: {config.result_dir}")
    
    # 打印汇总信息
    print("\n========== 最优层级汇总 ==========")
    for res in all_results:
        print(f"{res['var']}: 最优层级{res['best_level']} | MAE={res['best_mae']:.4f} | R²={res['best_r2']:.4f}")

# ===================== 8. 主执行流程 =====================
if __name__ == "__main__":
    batch_analyze()