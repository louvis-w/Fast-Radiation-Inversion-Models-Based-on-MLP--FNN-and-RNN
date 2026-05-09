import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 配置参数 =====================
class Config:
    ml_data_path="E:/ERA5-ml-0.25-all-20100201-0000.nc"       
    rad_data_path="E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"  
    # 可选：批量读取文件的目录（预留扩展）
    # ml_data_dir = "E:/ERA5-ml-data/"
    # rad_data_dir = "E:/ERA5-rad-data/"

    input_dim=324                # 输入维度（324个通道的亮温）
    output_dim=16                # 输出维度（16个大气参数）
    hidden_dims=[1024, 512, 256] # 隐藏层维度
    dropout_rate=0.2             # Dropout率
    lr=1e-4                      # 学习率
    weight_decay=1e-5            # 权重衰减
    epochs=100                   # 训练轮数
    batch_size=128               # 批次大小
    val_split=0.2                # 验证集比例
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

config=Config()

# ===================== 2. 数据读取与预处理 =====================
class ERA5Dataset(Dataset):
    def __init__(self, ml_data, rad_data, normalize=True):
        """
        初始化数据集
        :param ml_data: 大气参数数据 (numpy数组, shape=[N, 16])
        :param rad_data: 亮温数据 (numpy数组, shape=[N, 324])
        :param normalize: 是否归一化
        """
        self.normalize = normalize
        
        # 数据归一化
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

def load_single_file_data(ml_path, rad_path):
    """
    加载单个文件的数据
    :return: 展平后的大气参数数据、亮温数据
    """
    # 读取大气参数文件
    ml_ds = xr.open_dataset(ml_path)
    rad_ds = xr.open_dataset(rad_path)
    
    # 提取大气参数变量（按顺序）
    ml_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    
    # 数据预处理：选择level=1层（可根据需求调整），展平空间维度
    ml_data_list = []
    for var in ml_vars:
        # 处理缺失值（用均值填充）
        var_data = ml_ds[var].isel(time=0, level=0).values  # [721, 1440]
        var_data = np.nan_to_num(var_data, nan=np.nanmean(var_data))
        ml_data_list.append(var_data.flatten())
    
    # 合并大气参数: [721*1440, 16]
    ml_data = np.stack(ml_data_list, axis=1)
    
    # 提取亮温数据：选择time=0，展平空间维度 [324, 721, 1440] -> [721*1440, 324]
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    
    # 过滤无效数据
    valid_mask = ~(np.isnan(ml_data).any(axis=1) | np.isnan(rad_data).any(axis=1))
    ml_data = ml_data[valid_mask]
    rad_data = rad_data[valid_mask]
    
    return ml_data, rad_data

def load_batch_files_data(ml_dir, rad_dir):
    """
    批量加载多个文件的数据（预留扩展接口）
    :param ml_dir: 大气参数文件目录
    :param rad_dir: 亮温数据文件目录
    :return: 合并后的大气参数数据、亮温数据
    """
    ml_files = sorted([f for f in os.listdir(ml_dir) if f.endswith('.nc')])
    rad_files = sorted([f for f in os.listdir(rad_dir) if f.endswith('.nc')])
    
    all_ml_data = []
    all_rad_data = []
    
    for ml_file, rad_file in tqdm(zip(ml_files, rad_files), desc="Loading files"):
        ml_path = os.path.join(ml_dir, ml_file)
        rad_path = os.path.join(rad_dir, rad_file)
        
        ml_data, rad_data = load_single_file_data(ml_path, rad_path)
        all_ml_data.append(ml_data)
        all_rad_data.append(rad_data)
    
    return np.concatenate(all_ml_data, axis=0), np.concatenate(all_rad_data, axis=0)

# ===================== 3. MLP模型定义 =====================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ===================== 4. 训练与验证函数 =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """
    训练模型
    """
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for rad_batch, ml_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            rad_batch = rad_batch.to(device)
            ml_batch = ml_batch.to(device)
            
            # 前向传播
            outputs = model(rad_batch)
            loss = criterion(outputs, ml_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * rad_batch.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rad_batch, ml_batch in val_loader:
                rad_batch = rad_batch.to(device)
                ml_batch = ml_batch.to(device)
                
                outputs = model(rad_batch)
                loss = criterion(outputs, ml_batch)
                val_loss += loss.item() * rad_batch.size(0)
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mlp_model.pth")
        
        # 打印日志
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}, Best Val Loss = {best_val_loss:.6f}")
    
    return model, train_losses, val_losses

# ===================== 5. 主执行流程 =====================
if __name__ == "__main__":
    # 1. 加载数据
    print("Loading data...")
    # 加载单个文件（基础版本）
    ml_data, rad_data = load_single_file_data(config.ml_data_path, config.rad_data_path)
    
    # 可选：加载批量文件（取消注释启用）
    # ml_data, rad_data = load_batch_files_data(config.ml_data_dir, config.rad_data_dir)
    
    print(f"Data loaded: ML data shape = {ml_data.shape}, Rad data shape = {rad_data.shape}")
    
    # 2. 创建数据集和数据加载器
    dataset = ERA5Dataset(ml_data, rad_data)
    val_size = int(config.val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. 初始化模型、损失函数、优化器
    model = MLP(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    
    criterion = nn.MSELoss()  # 回归任务使用MSE损失
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # 4. 训练模型
    print("Starting training...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.epochs, config.device
    )
    
    # 5. 加载最佳模型并测试
    print("\nLoading best model...")
    model.load_state_dict(torch.load("best_mlp_model.pth"))
    model.eval()
    
    # 随机选择一个样本测试
    test_rad, test_ml = val_dataset[0]
    test_rad = test_rad.unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        pred_ml = model(test_rad)
    
    # 反归一化（如果启用了归一化）
    if dataset.normalize:
        test_ml = test_ml.cpu().numpy() * dataset.ml_std + dataset.ml_mean
        pred_ml = pred_ml.cpu().numpy().squeeze() * dataset.ml_std + dataset.ml_mean
    
    print("\nSample prediction:")
    print(f"True ML params (first 5): {test_ml[:5]}")
    print(f"Pred ML params (first 5): {pred_ml[:5]}")
    
    print("\nTraining completed! Best model saved as 'best_mlp_model.pth'")