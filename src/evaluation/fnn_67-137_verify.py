import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import glob
import traceback
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
import gc 

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (增加调试开关)
# ==========================================
CONFIG = {
    'rad_dir': '/array1/DataArchives/radiances/radiance/',
    'rad_pattern': 'ERA5-rad-0.25-*.nc', 
    'ml_dir': '/array1/DataArchives/radiances/era5_ml/',
    'ml_pattern': 'ERA5-ml-0.25-*.nc', 
    
    # 16 个目标变量 (根据实际文件调整！)
    'target_vars_list': [
        't', 'q', 'u', 'v', 'w', 
        'o3', 'clwc', 'ciwc', 'cc', 
        'z', 'vo', 'd', 'lnsp', 'skt', 'sp' 
    ],
    
    'input_dim': 324,      
    'lat_stride': 4,       # 纬度1°步长 (0.25°分辨率→4步)
    'lon_stride': 8,       # 经度2°步长 (0.25°分辨率→8步)
    'seq_length': 5,       # LSTM序列长度
    'hidden_size': 256,    # LSTM隐藏层维度
    'num_layers': 2,       # LSTM层数
    'batch_size': 2048,    # 批次大小
    'learning_rate': 0.001,# 学习率
    'epochs': 15,          # 训练轮数
    'train_split': 0.8,    # 训练/验证划分比例
    'level_pressure_path': None,  # 模型层气压数据路径 (None则从ml文件读取)
    'debug_mode': True     # 调试模式：打印详细信息
}

# ==========================================
# 2. 数据集 (增强错误调试 + 模型层气压映射)
# ==========================================
class GlobalUniversalDataset(Dataset):
    def __init__(self, config, current_target):
        self.config = config
        self.target_var = current_target
        self.debug = config['debug_mode']
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        # --------------------------
        # 第一步：检查文件是否存在
        # --------------------------
        self.rad_files = sorted(glob.glob(os.path.join(config['rad_dir'], config['rad_pattern'])))
        self.ml_files = sorted(glob.glob(os.path.join(config['ml_dir'], config['ml_pattern'])))
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"📌 目标变量: {current_target}")
            print(f"📂 辐射率文件路径: {config['rad_dir']}")
            print(f"📂 ERA5文件路径: {config['ml_dir']}")
            print(f"🔍 辐射率文件数: {len(self.rad_files)} | ERA5文件数: {len(self.ml_files)}")
        
        # 文件为空直接返回
        if len(self.rad_files) == 0 or len(self.ml_files) == 0:
            print(f"❌ 致命错误：文件列表为空！")
            if len(self.rad_files) == 0:
                print(f"   - 辐射率文件匹配规则: {config['rad_pattern']}")
            if len(self.ml_files) == 0:
                print(f"   - ERA5文件匹配规则: {config['ml_pattern']}")
            self.valid = False
            return
        
        # 限制文件数量 (调试用)
        min_len = min(len(self.rad_files), len(self.ml_files))
        self.rad_files = self.rad_files[:min_len]
        self.ml_files = self.ml_files[:min_len]
        
        if self.debug:
            print(f"🔧 实际读取文件数: {len(self.rad_files)}")
            print(f"📍 全球采样步长: 纬度{config['lat_stride']}步 | 经度{config['lon_stride']}步")
        
        # --------------------------
        # 初始化变量
        # --------------------------
        X_list, y_list, Aux_list = [], [], []
        self.coord_grid = None  # 经纬度网格
        self.valid = False      # 数据是否有效
        self.level_pressures = None  # 模型层→气压映射
        self.is_3d_var = False  # 是否为3D变量 (含模型层)
        self.real_output_dim = 0 # 输出维度
        
        # --------------------------
        # 第二步：批量读取文件
        # --------------------------
        for i, (f_rad, f_ml) in enumerate(zip(self.rad_files, self.ml_files)):
            try:
                if self.debug and i % 5 == 0:  # 每5个文件打印一次进度
                    print(f"\n🔄 处理第{i+1}/{len(self.rad_files)}组文件:")
                    print(f"   - 辐射率: {os.path.basename(f_rad)}")
                    print(f"   - ERA5: {os.path.basename(f_ml)}")
                
                # --- 1. 读取辐射率数据 (输入X) ---
                with xr.open_dataset(f_rad, engine='netcdf4') as ds_rad:
                    # 找到324维特征的变量
                    var_x = None
                    if 'Brightness_Temperature' in ds_rad.data_vars:
                        var_x = ds_rad['Brightness_Temperature']
                    else:
                        for v in ds_rad.data_vars:
                            if 324 in ds_rad[v].shape:
                                var_x = ds_rad[v]
                                break
                    
                    if var_x is None:
                        if self.debug:
                            print(f"⚠️  文件{f_rad}未找到324维特征变量，跳过")
                        continue
                    
                    # 自动识别经纬度维度名
                    d_lat, d_lon = self._get_lat_lon_dims(var_x.dims, ds_rad)
                    if self.debug and self.coord_grid is None:
                        print(f"📐 辐射率数据维度名: 纬度={d_lat} | 经度={d_lon}")
                    
                    # 构建经纬度网格 (仅首次执行)
                    if self.coord_grid is None:
                        self.coord_grid = self._build_coord_grid(ds_rad, d_lat, d_lon)
                        if self.debug:
                            print(f"🌍 全球采样点数量: {len(self.coord_grid)}")
                    
                    # 空间采样 (按步长抽取)
                    sel_dict = {d_lat: slice(0, None, config['lat_stride']), 
                                d_lon: slice(0, None, config['lon_stride'])}
                    raw_x = var_x.isel(**sel_dict).values
                    
                    # 4D兼容处理 (时间, 纬度, 经度, 特征)
                    if raw_x.ndim == 3:
                        raw_x = np.expand_dims(raw_x, axis=0)  # 3D→4D
                    
                    # 确保324维特征在最后一维
                    if 324 in raw_x.shape:
                        axis_ch = raw_x.shape.index(324)
                        raw_x = np.moveaxis(raw_x, axis_ch, -1)
                    
                    # 重塑为 (时间步, 空间点, 特征数)
                    if raw_x.ndim == 4:
                        T_dim = raw_x.shape[0]
                        data_x = raw_x.reshape(T_dim, -1, 324)
                    else:
                        if self.debug:
                            print(f"⚠️  辐射率数据维度异常: {raw_x.ndim}维，跳过")
                        continue
                
                # --- 2. 读取ERA5标签数据 (y) ---
                with xr.open_dataset(f_ml, engine='netcdf4') as ds_ml:
                    # 检查目标变量是否存在
                    if self.target_var not in ds_ml.data_vars:
                        if self.debug:
                            print(f"⚠️  文件{f_ml}未找到变量{self.target_var}，变量列表: {list(ds_ml.data_vars)[:10]}...")
                        continue
                    
                    target = ds_ml[self.target_var]
                    
                    # 读取模型层气压 (仅首次执行)
                    if self.level_pressures is None:
                        self.level_pressures = self._get_level_pressures(ds_ml)
                    
                    # 识别标签的经纬度维度名
                    t_lat, t_lon = self._get_lat_lon_dims(target.dims, ds_ml)
                    
                    # 空间采样
                    sel_dict = {t_lat: slice(0, None, config['lat_stride']), 
                                t_lon: slice(0, None, config['lon_stride'])}
                    raw_y = target.isel(**sel_dict).values
                    
                    # 维度兼容处理
                    if 'time' not in target.dims and raw_y.ndim >= 2:
                        raw_y = np.expand_dims(raw_y, axis=0)
                    if raw_y.ndim == 2:
                        raw_y = np.expand_dims(raw_y, axis=0)
                    
                    # 判断是否为3D变量 (含137层模型层)
                    self.is_3d_var = 137 in raw_y.shape
                    if self.is_3d_var:
                        axis_lev = raw_y.shape.index(137)
                        raw_y = np.moveaxis(raw_y, axis_lev, -1)
                        feat_dim = 137
                    else:
                        feat_dim = 1
                        if raw_y.ndim == 3:
                            raw_y = np.expand_dims(raw_y, axis=-1)
                    
                    # 重塑为 (时间步, 空间点, 特征数)
                    data_y = raw_y.reshape(raw_y.shape[0], -1, feat_dim)
                
                # --- 3. 检查维度匹配并保存 ---
                if data_x.shape[:2] == data_y.shape[:2]:
                    X_list.append(np.nan_to_num(data_x))
                    y_list.append(np.nan_to_num(data_y))
                    # 生成空间点索引
                    N_pts = data_x.shape[1]
                    Aux_list.append(np.tile(np.arange(N_pts), (data_x.shape[0], 1)))
                else:
                    if self.debug:
                        print(f"⚠️  维度不匹配: X={data_x.shape[:2]} | Y={data_y.shape[:2]}，跳过")
                
            except Exception as e:
                if self.debug:
                    print(f"❌ 处理文件失败: {e}")
                    print(traceback.format_exc())
                continue
        
        # --------------------------
        # 第三步：数据后处理
        # --------------------------
        if not X_list or not y_list:
            print(f"❌ {current_target} 无有效数据！")
            self.valid = False
            return
        
        # 合并数据
        self.raw_X = np.concatenate(X_list, axis=0)
        self.raw_Y = np.concatenate(y_list, axis=0)
        self.raw_Locs = np.concatenate(Aux_list, axis=0)
        
        # 基本信息
        self.valid = True
        self.real_output_dim = self.raw_Y.shape[-1]
        
        if self.debug:
            print(f"\n✅ 数据读取完成:")
            print(f"   - 时间步总数: {self.raw_X.shape[0]}")
            print(f"   - 空间点数量: {self.raw_X.shape[1]}")
            print(f"   - 输出维度: {self.real_output_dim} (3D变量: {self.is_3d_var})")
        
        # 数据归一化
        self.X_scaled = self.scaler_X.fit_transform(
            self.raw_X.reshape(-1, 324)
        ).reshape(self.raw_X.shape)
        
        Y_flat = self.raw_Y.reshape(-1, self.real_output_dim)
        self.y_scaled = self.scaler_y.fit_transform(Y_flat).reshape(self.raw_Y.shape)
        
        # 构建LSTM序列
        self.X_seq, self.y_seq, self.loc_seq = self._create_sequences(
            self.X_scaled, self.y_scaled, self.raw_Locs
        )
        
        if self.debug:
            print(f"🔗 LSTM序列构建完成: 总样本数={len(self.X_seq)}")
    
    def _get_lat_lon_dims(self, dims, ds):
        """自动识别经纬度维度名"""
        d_lat, d_lon = None, None
        # 优先匹配常见维度名
        lat_candidates = ['lat', 'latitude', 'y', 'lat_0', 'latitude_0']
        lon_candidates = ['lon', 'longitude', 'x', 'lon_0', 'longitude_0']
        
        for dim in dims:
            if dim in lat_candidates:
                d_lat = dim
            if dim in lon_candidates:
                d_lon = dim
        
        # 兜底：取倒数第二/第一维度
        if d_lat is None or d_lon is None:
            if self.debug:
                print(f"⚠️  未识别到经纬度维度名，使用兜底规则: {dims}")
            d_lat = dims[-2] if len(dims)>=2 else None
            d_lon = dims[-1] if len(dims)>=1 else None
        
        return d_lat, d_lon
    
    def _build_coord_grid(self, ds, d_lat, d_lon):
        """构建经纬度网格并采样"""
        # 获取原始经纬度
        if d_lat in ds.coords:
            raw_lats = ds[d_lat].values
        else:
            raw_lats = np.linspace(-90, 90, ds.sizes[d_lat])
        
        if d_lon in ds.coords:
            raw_lons = ds[d_lon].values
        else:
            raw_lons = np.linspace(0, 360, ds.sizes[d_lon])
        
        # 构建网格
        LATS, LONS = np.meshgrid(raw_lats, raw_lons, indexing='ij')
        
        # 按步长采样
        s_lat = self.config['lat_stride']
        s_lon = self.config['lon_stride']
        lats_sub = LATS[::s_lat, ::s_lon]
        lons_sub = LONS[::s_lat, ::s_lon]
        
        # 展平为 (N, 2) 数组
        return np.stack([lats_sub.flatten(), lons_sub.flatten()], axis=1)
    
    def _get_level_pressures(self, ds):
        """获取模型层对应的实际气压值 (hPa)"""
        if 'level_pressure' in ds.data_vars:
            # 转换为hPa (若原单位为Pa)
            pressures = ds['level_pressure'].values / 100
        elif 'pressure_levels' in ds.coords:
            pressures = ds['pressure_levels'].values
        elif self.config['level_pressure_path'] is not None:
            # 从外部文件读取
            level_ds = xr.open_dataset(self.config['level_pressure_path'])
            pressures = level_ds['level_pressure'].values / 100
        else:
            # 无气压数据，返回模型层索引
            pressures = np.arange(1, 138)
            if self.debug:
                print(f"⚠️  未找到气压数据，使用模型层索引 (1-137)")
        
        return pressures
    
    def _create_sequences(self, X, y, locs):
        """构建LSTM时序序列"""
        T, N, F = X.shape
        seq_len = self.config['seq_length']
        Xs, ys, ls = [], [], []
        
        for i in range(T - seq_len):
            # 序列维度: (空间点, 时间步, 特征)
            Xs.append(np.transpose(X[i:i+seq_len], (1, 0, 2)))
            ys.append(y[i+seq_len-1])
            ls.append(locs[i+seq_len-1])
        
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ls)
    
    def __len__(self):
        return len(self.X_seq) if self.valid else 0
    
    def __getitem__(self, idx):
        if not self.valid:
            raise ValueError("数据集无效！")
        return (
            torch.FloatTensor(self.X_seq[idx]),
            torch.FloatTensor(self.y_seq[idx]),
            self.loc_seq[idx]
        )

# ==========================================
# 3. LSTM模型
# ==========================================
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out

# ==========================================
# 4. 可视化函数 (修正气压标注)
# ==========================================
def plot_level_error(y_true, y_pred, var_name, dataset, target_level_idx=80):
    """绘制指定模型层的误差图 (标注实际气压)"""
    if not dataset.is_3d_var:
        print(f"ℹ️ {var_name}为2D变量，跳过层级误差分析")
        return
    
    # 获取层级标注
    if len(dataset.level_pressures) >= target_level_idx + 1:
        level_press = dataset.level_pressures[target_level_idx]
        if isinstance(level_press, (int, float)) and level_press <= 137:
            level_label = f"ModelLevel_{target_level_idx+1}({level_press:.1f}hPa)"
        else:
            level_label = f"ModelLevel_{target_level_idx+1}"
    else:
        level_label = f"ModelLevel_{target_level_idx+1}"
    
    # 计算误差
    t_true = y_true[:, target_level_idx]
    t_pred = y_pred[:, target_level_idx]
    mae = mean_absolute_error(t_true, t_pred)
    rmse = np.sqrt(np.mean((t_pred - t_true)**2))
    
    print(f"\n{'*'*60}")
    print(f"📊 {var_name} - {level_label} 误差分析")
    print(f"MAE (平均绝对误差): {mae:.4f} K")
    print(f"RMSE (均方根误差): {rmse:.4f} K")
    print(f"{'*'*60}")
    
    # 绘制误差图
    coords = dataset.coord_grid[dataset.loc_seq]
    lons = coords[:, 1]
    lats = coords[:, 0]
    error = t_pred - t_true
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    sc = ax.scatter(lons, lats, c=error, s=1, cmap='seismic', vmin=-5, vmax=5)
    ax.set_title(f"{var_name} - {level_label} Error (Pred-True) [K]", fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    plt.colorbar(sc, ax=ax, label='Error (K)')
    plt.tight_layout()
    plt.savefig(f"{var_name}_{level_label}_error.png", dpi=150)
    plt.close()

def plot_global_map(y_true, y_pred, var_name, dataset):
    """绘制全球地表/单层变量分布图"""
    dim = y_true.shape[1]
    
    # 确定绘制层级
    if dim > 1:
        target_idx = dim - 1  # 地表层
        if len(dataset.level_pressures) >= target_idx + 1:
            level_press = dataset.level_pressures[target_idx]
            level_name = f"NearSurface({level_press:.1f}hPa)"
        else:
            level_name = "NearSurface(ModelLevel_{target_idx+1})"
    else:
        target_idx = 0
        level_name = "SingleLevel"
    
    # 提取数据
    surf_true = y_true[:, target_idx]
    surf_pred = y_pred[:, target_idx]
    surf_err = surf_pred - surf_true
    
    # 获取经纬度
    coords = dataset.coord_grid[dataset.loc_seq]
    lons = coords[:, 1]
    lats = coords[:, 0]
    
    # 绘图
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))
    titles = [
        f"{var_name} - True ({level_name})",
        f"{var_name} - Pred ({level_name})",
        f"{var_name} - Error (Pred-True) ({level_name})"
    ]
    datas = [surf_true, surf_pred, surf_err]
    cmaps = ['jet', 'jet', 'seismic']
    vlims = [(None, None), (None, None), (-5, 5)]
    
    for ax, data, title, cmap, (vmin, vmax) in zip(axs, datas, titles, cmaps, vlims):
        sc = ax.scatter(lons, lats, c=data, s=1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        plt.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f"GlobalMap_{var_name}_{level_name}.png", dpi=150)
    plt.close()

# ==========================================
# 5. 主训练流程
# ==========================================
def run_pipeline():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 训练开始 | 设备: {device}")
    print(f"📝 配置信息: {CONFIG}")
    
    # 遍历所有目标变量
    for var in CONFIG['target_vars_list']:
        print(f"\n{'='*80}")
        print(f"🎯 开始处理变量: {var}")
        print(f"{'='*80}")
        
        # 初始化数据集
        dataset = GlobalUniversalDataset(CONFIG, var)
        if not dataset.valid:
            print(f"🚫 跳过变量 {var} (数据集无效)")
            continue
        
        # 划分训练/验证集
        train_size = int(len(dataset) * CONFIG['train_split'])
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # 构建数据加载器
        train_loader = DataLoader(
            train_ds, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True,
            num_workers=0,  # 避免多进程问题
            pin_memory=True if device.type == 'cuda' else False
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # 初始化模型
        model = LSTM_Model(
            input_size=CONFIG['input_dim'],
            hidden_size=CONFIG['hidden_size'],
            output_size=dataset.real_output_dim,
            num_layers=CONFIG['num_layers']
        ).to(device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=CONFIG['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # 训练模型
        print(f"\n🔥 开始训练 {var} | 训练样本: {len(train_ds)} | 验证样本: {len(test_ds)}")
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (bx, by, _) in enumerate(train_loader):
                # 数据移到设备
                bx = bx.to(device)
                by = by.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 累计损失
                train_loss += loss.item() * bx.size(0)
                
                # 打印进度
                if CONFIG['debug_mode'] and batch_idx % 10 == 0:
                    print(f"   Epoch {epoch+1}/{CONFIG['epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_ds)
            print(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} | 平均训练损失: {avg_train_loss:.6f}")
        
        # 验证模型
        print(f"\n📈 开始验证 {var}")
        model.eval()
        preds, trues, locs = [], [], []
        
        with torch.no_grad():
            for bx, by, loc_idx in test_loader:
                bx = bx.to(device)
                # 预测
                pred = model(bx)
                # 保存结果
                preds.append(pred.cpu().numpy())
                trues.append(by.cpu().numpy())
                locs.append(loc_idx.numpy())
        
        # 合并结果
        preds_real = dataset.scaler_y.inverse_transform(
            np.concatenate(preds).reshape(-1, dataset.real_output_dim)
        ).reshape(-1, dataset.real_output_dim)
        trues_real = dataset.scaler_y.inverse_transform(
            np.concatenate(trues).reshape(-1, dataset.real_output_dim)
        ).reshape(-1, dataset.real_output_dim)
        
        # 保存结果
        save_path = f"results_{var}.npz"
        np.savez_compressed(
            save_path,
            pred=preds_real,
            true=trues_real,
            loc_idx=np.concatenate(locs),
            coords=dataset.coord_grid,
            level_pressures=dataset.level_pressures
        )
        print(f"💾 结果已保存至: {save_path}")
        
        # 可视化
        if var == 't':  # 温度变量分析Level 80
            plot_level_error(trues_real, preds_real, var, dataset, target_level_idx=80)
        # 绘制全球地图
        plot_global_map(trues_real, preds_real, var, dataset)
        
        # 清理内存
        print(f"\n🧹 清理内存 | 变量: {var}")
        del model, dataset, train_loader, test_loader, train_ds, test_ds
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n🎉 所有变量处理完成！")

# ==========================================
# 入口函数
# ==========================================
if __name__ == "__main__":
    run_pipeline()