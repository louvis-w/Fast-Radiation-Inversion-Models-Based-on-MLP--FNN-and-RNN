import os
import time
import psutil
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 配置参数（仅u/v/w/t） =====================
class Config:
    ml_data_path = "E:/ERA5-ml-0.25-all-20100201-0000.nc"
    rad_data_path = "E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"
    n_levels = 80
    input_dim = 324
    task_vars = ['t']  # 先测试温度t
    model_prefix = "best_mlp_80level_"
    model_dir = "D:/Desktop/py库"
    target_level = 79
    fig_size = (16, 8)
    dpi = 300
    cmap = 'plasma'  # 温度常用配色（更直观）
    error_cmap = 'coolwarm'
    result_dir = "global_80level_analysis_fixed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    # 关键：设置中文字体（解决乱码）
    font = {'family': 'SimHei', 'size': 10}
    plt.rcParams.update({'font.family': font['family'], 'font.size': font['size']})
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

config = Config()

# ===================== 2. 数据集类 =====================
class ERA5SingleTaskDataset(Dataset):
    def __init__(self, ml_data, rad_data, lon, lat, task_var):
        self.task_var = task_var
        self.var_idx = ['crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
                        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'].index(task_var)
        self.lon = lon
        self.lat = lat
        self.grid_shape = (len(lat), len(lon))
        self.total_grid = len(lat) * len(lon)
        
        # 有效网格索引
        self.rad_valid = ~np.all(np.isnan(rad_data), axis=1)
        self.valid_ml_data = ml_data[self.rad_valid]
        self.valid_rad_data = rad_data[self.rad_valid]
        
        # 裁剪80层数据
        self.valid_ml_data = self.valid_ml_data[:, :config.n_levels * 16]
        self.task_ml_data = self.valid_ml_data[:, self.var_idx::16]
        
        # 归一化（复用训练逻辑）
        self.ml_min = np.min(self.task_ml_data, axis=0)
        self.ml_max = np.max(self.task_ml_data, axis=0)
        self.task_ml_data = 2 * (self.task_ml_data - self.ml_min) / (self.ml_max - self.ml_min + 1e-8) - 1
        self.valid_rad_data = np.clip(rad_data, -1e4, 1e4)
        self.rad_min = np.min(self.valid_rad_data, axis=0)
        self.rad_max = np.max(self.valid_rad_data, axis=0)
        self.valid_rad_data = 2 * (self.valid_rad_data - self.rad_min) / (self.rad_max - self.rad_min + 1e-8) - 1
        
        self.rad_data = torch.FloatTensor(self.valid_rad_data)
        self.task_ml_data = torch.FloatTensor(self.task_ml_data)
    
    def __len__(self):
        return len(self.rad_data)
    
    def __getitem__(self, idx):
        return self.rad_data[idx], self.task_ml_data[idx]
    
    def inverse_normalize(self, normalized_data):
        return (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min
    
    def restore_grid(self, flat_data):
        grid_data = np.full(self.total_grid, np.nan)
        grid_data[self.rad_valid] = flat_data
        return grid_data.reshape(self.grid_shape)

# ===================== 3. 模型+数据加载 =====================
class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(dim)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

def load_data():
    ml_ds = xr.open_dataset(config.ml_data_path)
    rad_ds = xr.open_dataset(config.rad_data_path)
    lon = ml_ds.longitude.values
    lat = ml_ds.latitude.values
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values.reshape(-1, 324)
    ml_vars = ['crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
               'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc']
    ml_data_list = []
    for level in range(config.n_levels):
        level_data = [ml_ds[var].isel(time=0, level=level).values.flatten() for var in ml_vars]
        ml_data_list.append(np.stack(level_data, axis=1))
    ml_flat = np.concatenate(ml_data_list, axis=1)
    return ml_flat, rad_data, lon, lat

# ===================== 4. 修复后的绘图函数（正常图例+中文） =====================
def plot_fixed_map(var_name, true_grid, pred_grid, error_grid, lon, lat, used_level):
    os.makedirs(os.path.join(config.result_dir, var_name), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=config.fig_size, dpi=config.dpi,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"{var_name}（温度）- 第{used_level+1}层全球分布", fontsize=14)

    # 1. 真实值（温度）
    ax1 = axes[0]
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.2)
    true_valid = true_grid[~np.isnan(true_grid)]
    levels_true = np.linspace(true_valid.min(), true_valid.max(), 12)
    im1 = ax1.contourf(lon, lat, true_grid, cmap=config.cmap, levels=levels_true, extend='both')
    ax1.set_title("真实温度分布")
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.05)
    cbar1.set_label("温度（K）", fontsize=9)  # 图例标签：温度+单位

    # 2. 预测值（温度）
    ax2 = axes[1]
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.2)
    pred_valid = pred_grid[~np.isnan(pred_grid)]
    levels_pred = np.linspace(pred_valid.min(), pred_valid.max(), 12)
    im2 = ax2.contourf(lon, lat, pred_grid, cmap=config.cmap, levels=levels_pred, extend='both')
    ax2.set_title("预测温度分布")
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.05)
    cbar2.set_label("温度（K）", fontsize=9)

    # 3. 误差（预测-真实）
    ax3 = axes[2]
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.2)
    error_valid = error_grid[~np.isnan(error_grid)]
    vmax = np.nanmax(np.abs(error_valid))
    levels_error = np.linspace(-vmax, vmax, 12)
    im3 = ax3.contourf(lon, lat, error_grid, cmap=config.error_cmap, levels=levels_error, extend='both')
    ax3.set_title("温度反演误差")
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.05)
    cbar3.set_label("误差（K）", fontsize=9)  # 图例标签：误差+单位

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(config.result_dir, var_name, f"{var_name}_level{used_level+1}_fixed.png")
    plt.savefig(fig_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()
    print(f"修复后的图已保存：{fig_path}")

# ===================== 5. 主流程 =====================
def main():
    ml_flat, rad_data, lon, lat = load_data()
    dataset = ERA5SingleTaskDataset(ml_flat, rad_data, lon, lat, 't')
    
    # 加载模型
    model = SingleTaskMLP(324, 80, [1024, 512, 256]).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, "best_mlp_80level_t.pth"), 
                                     map_location=config.device)['model_state_dict'])
    model.eval()

    # 预测
    dataloader = DataLoader(dataset, batch_size=config.batch_size)
    all_pred, all_true = [], []
    with torch.no_grad():
        for rad, ml in dataloader:
            pred = model(rad.to(config.device)).cpu().numpy()
            all_pred.append(dataset.inverse_normalize(pred))
            all_true.append(dataset.inverse_normalize(ml.numpy()))
    all_pred = np.concatenate(all_pred, axis=0)[:, config.target_level]
    all_true = np.concatenate(all_true, axis=0)[:, config.target_level]

    # 绘图
    true_grid = dataset.restore_grid(all_true)
    pred_grid = dataset.restore_grid(all_pred)
    plot_fixed_map('t', true_grid, pred_grid, pred_grid-true_grid, lon, lat, config.target_level)

if __name__ == "__main__":
    main()