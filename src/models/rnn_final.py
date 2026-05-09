import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg') # 服务器专用：后台画图，不弹窗
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 全局配置 (请确认路径)
# ==========================================
CONFIG = {
    # 输入：卫星观测 (Radiance)
    'rad_dir': '/array1/DataArchives/radiances/radiance/',
    'rad_pattern': 'ERA5-rad-0.25-*.nc', 
    
    # 标签：大气参数 (ERA5 ML Data)
    'ml_dir': '/array1/DataArchives/radiances/era5_ml/',
    'ml_pattern': 'ERA5-ml-0.25-*.nc', 
    
    # 任务参数
    'target_var': 't',     # 反演温度 t
    'input_dim': 324,      # 卫星有 324 个通道
    'output_dim': 80,      # 大气有 80 个垂直层
    
    # RNN 核心参数
    'seq_length': 5,       # 【关键】看过去 5 个时间点的数据
    'hidden_size': 256,    # LSTM 记忆容量
    'num_layers': 2,       # LSTM 层数
    
    # 训练参数
    'batch_size': 32,      # 因为序列数据比单点大，Batch设小一点稳妥
    'learning_rate': 0.001,
    'epochs': 100,         # 增加轮数，让它充分学习
    'train_split': 0.8
}

# ==========================================
# 2. 数据处理类 (修复版：自动识别维度名称)
# ==========================================
class FullSequenceDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        # --- A. 扫描并配对文件 ---
        rad_files = sorted(glob.glob(os.path.join(config['rad_dir'], config['rad_pattern'])))
        ml_files = sorted(glob.glob(os.path.join(config['ml_dir'], config['ml_pattern'])))
        
        min_len = min(len(rad_files), len(ml_files))
        if min_len == 0: raise FileNotFoundError("未找到任何匹配的文件！")
        
        rad_files = rad_files[:min_len]
        ml_files = ml_files[:min_len]
        
        print(f"检测到 {min_len} 对文件。正在构建时间序列数据库...")
        
        X_list, y_list = [], []
        
        # --- B. 循环读取所有文件 ---
        for i, (f_rad, f_ml) in enumerate(zip(rad_files, ml_files)):
            try:
                # 1. 读取卫星数据 (Input)
                with xr.open_dataset(f_rad) as ds:
                    var_x = None
                    if 'Brightness_Temperature' in ds: var_x = ds['Brightness_Temperature']
                    else:
                        for v in ds.data_vars:
                            if len(ds[v].shape) >= 2 and ds[v].shape[1] == 324:
                                var_x = ds[v]; break
                    
                    if var_x is None: continue
                    
                    # === 修复点 1：自动判断卫星数据的维度名 ===
                    # 优先找 x/y，找不到就找 lon/lat，再找不到找 longitude/latitude
                    dims = ds.dims
                    if 'x' in dims and 'y' in dims:
                        idx_dict = {'x': 20, 'y': 20}
                    elif 'lon' in dims and 'lat' in dims:
                        idx_dict = {'lon': 20, 'lat': 20}
                    elif 'longitude' in dims and 'latitude' in dims:
                        idx_dict = {'longitude': 20, 'latitude': 20}
                    else:
                        # 实在找不到，可能是 full disk，尝试按位置索引（第20行第20列）
                        # 这种写法不依赖名字，直接取第2和第3维度的第20个
                        # 假设形状是 (time, ch, y, x) -> 取 (:, :, 20, 20)
                        data_x = var_x[:, :, 20, 20].values
                        if data_x.ndim == 1: data_x = data_x.reshape(1, -1)
                        # 跳过下面的 .isel，直接进入下一步
                        pass 
                    
                    # 如果刚才没读（还在idx_dict阶段），则在这里读
                    if 'idx_dict' in locals():
                        data_x = var_x.isel(**idx_dict).values
                        if data_x.ndim == 1: data_x = data_x.reshape(1, -1)
                        del idx_dict # 清理变量以免污染下一次循环

                # 2. 读取 ERA5 标签 (Label)
                with xr.open_dataset(f_ml) as ds:
                    if config['target_var'] not in ds: continue
                    
                    # === 修复点 2：自动判断 ERA5 数据的维度名 ===
                    dims = ds.dims
                    target = ds[config['target_var']]
                    
                    # 准备切片字典，先把 level 放进去
                    # level 名字也可能是 level, lev, bottom_top 等
                    lev_name = 'level'
                    if 'lev' in dims: lev_name = 'lev'
                    
                    sel_dict = {lev_name: slice(0, 80)} # 取前80层
                    
                    # 再加经纬度
                    if 'x' in dims and 'y' in dims:
                        sel_dict.update({'x': 20, 'y': 20})
                    elif 'lon' in dims and 'lat' in dims:
                        sel_dict.update({'lon': 20, 'lat': 20})
                    elif 'longitude' in dims and 'latitude' in dims:
                        sel_dict.update({'longitude': 20, 'latitude': 20})
                    
                    data_y = target.isel(**sel_dict).values
                    if data_y.ndim == 1: data_y = data_y.reshape(1, -1)
                
                # 3. 存入列表
                # 确保两个都读到了才存
                if data_x.shape[0] == data_y.shape[0]:
                    X_list.append(data_x)
                    y_list.append(data_y)

            except Exception as e:
                # 打印详细错误方便调试
                # print(f"读取第 {i} 个文件出错: {e}")
                continue
            
            if (i+1) % 20 == 0: 
                print(f"已处理 {i+1}/{min_len} 个时间点...")
        
        # --- C. 拼接 ---
        if not X_list:
            raise ValueError("所有文件读取失败！请检查文件名或维度名称。")

        self.raw_X = np.concatenate(X_list, axis=0)
        self.raw_Y = np.concatenate(y_list, axis=0)
        
        print(f"\n原始数据加载完毕！Time Steps: {self.raw_X.shape[0]}")
        
        # --- D. 归一化 ---
        self.X_scaled = self.scaler_X.fit_transform(self.raw_X)
        self.y_scaled = self.scaler_y.fit_transform(self.raw_Y)
        
        # --- E. 序列化 ---
        self.X_seq, self.y_seq = self._create_sequences(self.X_scaled, self.y_scaled)
        print(f"序列构建完成，样本数: {len(self.X_seq)}")
        
    def _create_sequences(self, X, y):
        Xs, ys = [], []
        seq_len = self.config['seq_length']
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len - 1]) 
        return np.array(Xs), np.array(ys)

    def __len__(self): return len(self.X_seq)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X_seq[idx]), torch.FloatTensor(self.y_seq[idx])

# ==========================================
# 3. LSTM 反演模型
# ==========================================
class LSTM_Inversion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Inversion, self).__init__()
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        
        # 全连接层 (从 Hidden State -> 80层大气参数)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # x: (Batch, Seq, 324)
        out, _ = self.lstm(x)
        
        # 取最后一个时间步 (包含了整个序列的记忆)
        last_step = out[:, -1, :] 
        
        # 解码
        pred = self.fc(last_step)
        return pred

# ==========================================
# 4. 专业绘图函数 (6合1)
# ==========================================
def plot_analysis(y_true, y_pred):
    # 选取第40层 (中间层) 进行散点分析
    level_idx = 40 
    
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true[:, level_idx], y_pred[:, level_idx])
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f"RNN Inversion Analysis (Target: {CONFIG['target_var']}) - MAE: {mae:.4f}", fontsize=15)
    
    # 1. 散点图
    ax = axs[0, 0]
    ax.scatter(y_true[:, level_idx], y_pred[:, level_idx], s=10, alpha=0.6, c='orange')
    mi, ma = y_true[:, level_idx].min(), y_true[:, level_idx].max()
    ax.plot([mi, ma], [mi, ma], 'r--', lw=2, label='1:1 Line')
    ax.set_title(f"Level {level_idx} Scatter (R²={r2:.3f})")
    ax.set_xlabel("True Value"); ax.set_ylabel("Pred Value")
    ax.legend(); ax.grid(alpha=0.3)
    
    # 2. 误差分布直方图
    ax = axs[0, 1]
    err = y_pred[:, level_idx] - y_true[:, level_idx]
    ax.hist(err, bins=30, color='skyblue', edgecolor='k', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title(f"Level {level_idx} Error Dist")
    ax.grid(alpha=0.3)
    
    # 3. 垂直层 MAE 分布
    ax = axs[0, 2]
    mae_levels = np.mean(np.abs(y_pred - y_true), axis=0)
    ax.plot(np.arange(80), mae_levels, 'g-', lw=2, marker='o', markersize=3)
    ax.set_title("Vertical MAE Profile")
    ax.set_xlabel("Level (0-79)"); ax.set_ylabel("MAE")
    ax.grid(alpha=0.3)
    
    # 4. 概率密度对比 (PDF)
    ax = axs[1, 0]
    ax.hist(y_true[:, level_idx], bins=30, density=True, alpha=0.3, color='b', label='True')
    ax.hist(y_pred[:, level_idx], bins=30, density=True, alpha=0.3, color='orange', label='Pred')
    ax.set_title(f"Level {level_idx} Distribution")
    ax.legend(); ax.grid(alpha=0.3)
    
    # 5. 累积误差分布 (CDF)
    ax = axs[1, 1]
    sorted_err = np.sort(err)
    p = np.arange(len(err)) / (len(err)-1)
    ax.plot(sorted_err, p, 'purple', lw=2)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_title("Error CDF")
    ax.grid(alpha=0.3)
    
    # 6. 单样本廓线对比
    ax = axs[1, 2]
    # 随机取一个测试样本
    idx = np.random.randint(0, len(y_true))
    ax.plot(y_true[idx], np.arange(80), 'b-', lw=2, label='True')
    ax.plot(y_pred[idx], np.arange(80), 'r--', lw=2, label='RNN Pred')
    ax.invert_yaxis() # 0层通常在地面或顶层，根据习惯翻转
    ax.set_title(f"Single Profile (Sample {idx})")
    ax.set_xlabel("Value"); ax.set_ylabel("Level")
    ax.legend(); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_full_analysis.png')
    print("\n>>> 结果图已保存为: rnn_full_analysis.png")

# ==========================================
# 5. 训练主流程
# ==========================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载数据
    dataset = FullSequenceDataset(CONFIG)
    
    # 2. 划分数据集 (时间序列通常建议按顺序划分，这里简单随机划分)
    train_size = int(len(dataset) * CONFIG['train_split'])
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 3. 初始化模型
    model = LSTM_Inversion(CONFIG['input_dim'], CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['output_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    # 4. 训练
    print("\n开始训练 RNN 模型...")
    loss_history = []
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.6f}")
            
    # 5. 测试与绘图
    model.eval()
    preds, trues = [], []
    print("\n正在生成测试结果...")
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            preds.append(out.cpu().numpy())
            trues.append(by.numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反归一化
    preds_real = dataset.scaler_y.inverse_transform(preds)
    trues_real = dataset.scaler_y.inverse_transform(trues)
    
    # 画图
    plot_analysis(trues_real, preds_real)

if __name__ == "__main__":
    train_model()