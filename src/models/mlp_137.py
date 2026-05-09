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
    # 数据路径
    ml_data_path="E:/ERA5-ml-0.25-all-20100201-0000.nc"       
    rad_data_path="E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"  

    # 137层配置
    n_levels = 137
    input_dim = 324
    output_dim = n_levels * 16  # 2192维输出
    
    # 扩大模型参数配置（更深更宽）
    hidden_dims = [4096, 2048, 1024, 512]  # 模型参数翻倍
    dropout_rate = 0.1  # 降低dropout保留信息
    batch_norm_eps = 1e-5  # 提升batchnorm稳定性
    
    # 训练参数（80%训练/20%验证）
    lr = 3e-6  # 极低学习率避免梯度爆炸
    weight_decay = 1e-4  # 权重衰减抑制过拟合
    epochs = 50
    batch_size = 16  # 小批次适配高维输出
    train_ratio = 0.8  # 80%数据作为训练集
    val_ratio = 0.2    # 20%数据作为验证集
    grad_clip_norm = 1.0  # 梯度裁剪阈值
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ===================== 2. 数据集定义（彻底解决NaN） =====================
class ERA5Dataset(Dataset):
    def __init__(self, ml_data, rad_data, normalize=True):
        self.normalize = normalize
        
        # 第一步：全局数值裁剪（限制极端值）
        ml_data = np.clip(ml_data, -1e4, 1e4)
        rad_data = np.clip(rad_data, -1e4, 1e4)
        
        # 第二步：鲁棒最大最小归一化（避免除0）
        if normalize:
            # 亮温归一化到[-1, 1]
            self.rad_min = np.nanmin(rad_data, axis=0)
            self.rad_max = np.nanmax(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            # 137层参数归一化到[-1, 1]
            self.ml_min = np.nanmin(ml_data, axis=0)
            self.ml_max = np.nanmax(ml_data, axis=0)
            equal_mask = self.ml_max == self.ml_min
            self.ml_max[equal_mask] = self.ml_min[equal_mask] + 1e-8
            ml_data = 2 * (ml_data - self.ml_min) / (self.ml_max - self.ml_min) - 1
        
        # 第三步：最终裁剪+NaN/Inf填充
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

# ===================== 3. 137层数据加载函数 =====================
def load_137level_data(ml_path, rad_path):
    """加载137层全量数据，最大化有效样本"""
    ml_ds = xr.open_dataset(ml_path)
    rad_ds = xr.open_dataset(rad_path)
    
    ml_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    
    # 加载亮温数据（仅过滤全NaN样本）
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    rad_valid = ~np.all(np.isnan(rad_data), axis=1)
    rad_data = rad_data[rad_valid]
    
    # 加载137层大气参数
    ml_data_list = []
    max_samples = 80000  # 扩大样本量（适配80%训练）
    for level in range(config.n_levels):
        level_data = []
        for var in ml_vars:
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 内存安全裁剪
    if len(ml_data) > max_samples:
        idx = np.random.choice(len(ml_data), max_samples, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    # 打印数据统计
    print(f"数据加载完成 - 总样本数: {len(ml_data)}")
    print(f"大气参数维度: {ml_data.shape}, 亮温维度: {rad_data.shape}")
    print(f"参数数值范围: [{np.min(ml_data):.4f}, {np.max(ml_data):.4f}]")
    return ml_data, rad_data

# ===================== 4. 大参数MLP模型 =====================
class LargeMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(LargeMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建扩大版隐藏层（更多参数）
        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim, eps=bn_eps),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate) if i < len(hidden_dims)-1 else nn.Identity()
            ])
            prev_dim = dim
        
        # 输出层+Tanh约束输出范围
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        # 自定义权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)

# ===================== 5. 训练函数（梯度稳定+80/20划分） =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(epochs):
        # 训练阶段（80%数据）
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for rad_batch, ml_batch in pbar:
            rad_batch = rad_batch.to(device, non_blocking=True)
            ml_batch = ml_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 混合精度训练+梯度裁剪
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(rad_batch)
                    loss = criterion(outputs, ml_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(rad_batch)
                loss = criterion(outputs, ml_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
            
            # 过滤NaN损失
            batch_loss = loss.item() if not torch.isnan(loss) else 0.0
            train_loss += batch_loss * rad_batch.size(0)
            avg_loss = train_loss / ((pbar.n + 1) * config.batch_size)
            pbar.set_postfix({"Train Loss": f"{avg_loss:.6f}"})
        
        # 验证阶段（20%数据）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rad_batch, ml_batch in val_loader:
                rad_batch = rad_batch.to(device, non_blocking=True)
                ml_batch = ml_batch.to(device, non_blocking=True)
                
                outputs = model(rad_batch)
                loss = criterion(outputs, ml_batch)
                batch_loss = loss.item() if not torch.isnan(loss) else 0.0
                val_loss += batch_loss * rad_batch.size(0)
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 保存最佳模型
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mlp_137level_large.pth")
        
        # 打印日志
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss (80% data): {avg_train_loss:.6f} | Val Loss (20% data): {avg_val_loss:.6f}")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print("-" * 60)
    
    return model, train_losses, val_losses

# ===================== 6. 主执行流程 =====================
if __name__ == "__main__":
    # 1. 加载137层数据
    print("加载137层大气参数数据...")
    ml_data, rad_data = load_137level_data(config.ml_data_path, config.rad_data_path)
    
    # 2. 创建数据集（80%训练/20%验证）
    dataset = ERA5Dataset(ml_data, rad_data)
    total_samples = len(dataset)
    train_size = int(config.train_ratio * total_samples)
    val_size = total_samples - train_size
    
    # 固定随机种子，保证划分一致
    torch.manual_seed(42)
    np.random.seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n数据集划分:")
    print(f"训练集（80%）: {len(train_dataset)} samples")
    print(f"验证集（20%）: {len(val_dataset)} samples")
    
    # 3. 数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size * 2, 
        shuffle=False,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    # 4. 初始化大参数模型
    print("\n初始化大参数MLP模型...")
    model = LargeMLP(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
        bn_eps=config.batch_norm_eps
    ).to(config.device)
    
    # 打印模型参数规模
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params/1e6:.2f}M (扩大版)")
    
    # 5. 定义损失和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 6. 启动训练
    print("\n启动137层MLP训练（80%数据训练）...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config.epochs, config.device
    )
    
    # 7. 加载最佳模型
    print("\n训练完成！加载最佳模型...")
    model.load_state_dict(torch.load("best_mlp_137level_large.pth"))
    model.eval()
    
    # 验证样本预测
    test_rad, test_ml = val_dataset[0]
    test_rad = test_rad.unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        pred_ml = model(test_rad)
    
    print(f"\n最佳模型保存路径: best_mlp_137level_large.pth")
    print(f"预测输出维度: {pred_ml.shape} (匹配137×16=2192维)")