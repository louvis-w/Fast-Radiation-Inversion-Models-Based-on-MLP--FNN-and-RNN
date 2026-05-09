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

# ===================== 1. 配置参数（中下层80层+16任务） =====================
class Config:
    # 数据路径
    ml_data_path="E:/ERA5-ml-0.25-all-20100201-0000.nc"       
    rad_data_path="E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"  

    # 中下层80层配置（取最后80层，对应250~1000hPa，0~10km）
    n_levels = 80  # 保留80层，但改为中下层
    input_dim = 324
    # 16个独立任务（每个任务输出80层对应参数）
    task_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    task_output_dim = n_levels  # 每个任务输出80维（中下层80层）
    
    # 轻量化模型配置（适配单任务）
    hidden_dims = [1024, 512, 256]
    dropout_rate = 0.1
    batch_norm_eps = 1e-5
    
    # 训练参数
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 50
    batch_size = 32
    train_ratio = 0.8
    val_ratio = 0.2
    grad_clip_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 任务训练顺序（优先训练核心参数如t/q）
    train_task_order = ['t', 'q', 'u', 'v', 'z', 'lnsp', 'o3', 
                        'crwc', 'cswc', 'etadot', 'vo', 'd', 
                        'clwc', 'ciwc', 'cc', 'w']

config = Config()

# ===================== 2. 单任务数据集（中下层80层+独立参数） =====================
class ERA5SingleTaskDataset(Dataset):
    def __init__(self, ml_data, rad_data, task_var, normalize=True):
        self.normalize = normalize
        self.task_var = task_var
        self.var_idx = config.task_vars.index(task_var)  # 当前任务参数索引
        self.n_levels = config.n_levels
        
        # 第一步：裁剪为中下层80层数据（已在load函数中处理，此处直接提取）
        ml_data = ml_data[:, :config.n_levels * len(config.task_vars)]
        # 提取当前任务的80层数据
        self.task_ml_data = ml_data[:, self.var_idx::len(config.task_vars)]  # 步长16，取对应参数
        
        # 第二步：数据预处理（仅针对当前任务）
        rad_data = np.clip(rad_data, -1e4, 1e4)
        self.task_ml_data = np.clip(self.task_ml_data, -1e4, 1e4)
        
        # 第三步：NaN填充（按任务单独填充）
        rad_data = np.nan_to_num(rad_data, nan=0.0, posinf=1e4, neginf=-1e4)
        self.task_ml_data = np.nan_to_num(self.task_ml_data, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 第四步：单独归一化（避免跨参数干扰）
        if normalize:
            # 亮温归一化
            self.rad_min = np.min(rad_data, axis=0)
            self.rad_max = np.max(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            # 当前任务参数归一化
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
        """当前任务参数反归一化"""
        if not self.normalize:
            return normalized_data
        return (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min

# ===================== 3. 中下层80层数据加载函数（核心修改） =====================
def load_80level_data(ml_path, rad_path):
    """加载中下层80层数据（最后80层，250~1000hPa），适配单任务训练"""
    ml_ds = xr.open_dataset(ml_path)
    rad_ds = xr.open_dataset(rad_path)
    
    # ===== 核心修改：截取中下层80层 =====
    total_levels = ml_ds.dims["level"]  # ERA5原始137层
    start_level = total_levels - config.n_levels  # 137-80=57，从第57层开始取最后80层
    end_level = total_levels  # 到137层结束
    selected_level_indices = list(range(start_level, end_level))  # 57~136层（中下层）
    
    # 验证层级对应的气压（关键！打印确认）
    all_pressures = ml_ds["level"].values
    selected_pressures = all_pressures[selected_level_indices]
    print("===== 中下层80层验证 =====")
    print(f"原始总层数: {total_levels}")
    print(f"截取层级索引: {start_level} ~ {end_level-1}")
    print(f"对应气压范围: {selected_pressures.min()} ~ {selected_pressures.max()} hPa")
    print(f"近地面层气压: {selected_pressures[-1]} hPa（第80层）")
    print(f"对流层中层气压: {selected_pressures[40]} hPa（第40层）")
    
    # 加载亮温数据（过滤全NaN样本）
    rad_data = rad_ds['Brightness_Temperature'].isel(time=0).values
    rad_data = np.transpose(rad_data, (1, 2, 0)).reshape(-1, 324)
    rad_valid = ~np.all(np.isnan(rad_data), axis=1)
    rad_data = rad_data[rad_valid]
    
    # 加载中下层80层大气参数（57~136层）
    ml_vars = config.task_vars
    ml_data_list = []
    for level in selected_level_indices:  # 取最后80层（中下层）
        level_data = []
        for var in ml_vars:
            # 提取当前层、当前参数的数据，并过滤无效样本
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 内存安全裁剪（保留8万样本）
    max_samples = 80000
    if len(ml_data) > max_samples:
        idx = np.random.choice(len(ml_data), max_samples, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    # 打印中下层数据统计
    print(f"\n中下层80层数据加载完成 - 总样本数: {len(ml_data)}")
    print(f"中下层参数维度: {ml_data.shape}, 亮温维度: {rad_data.shape}")
    return ml_data, rad_data

# ===================== 4. 单任务轻量化MLP模型（无修改） =====================
class SingleTaskMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(SingleTaskMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建轻量化隐藏层
        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim, eps=bn_eps),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout_rate) if i < len(hidden_dims)-1 else nn.Identity()
            ])
            prev_dim = dim
        
        # 输出层（仅预测当前任务的80层参数）
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        # 自定义初始化（适配单任务）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.1))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.model(x)

# ===================== 5. 单任务训练函数（无修改） =====================
def train_single_task(task_var, ml_data, rad_data):
    """训练单个参数的中下层80层预测模型"""
    # 1. 创建单任务数据集
    dataset = ERA5SingleTaskDataset(ml_data, rad_data, task_var)
    total_samples = len(dataset)
    train_size = int(config.train_ratio * total_samples)
    val_size = total_samples - train_size
    
    torch.manual_seed(42)
    np.random.seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n===== 训练任务: {task_var} (中下层80层) =====")
    print(f"训练集: {len(train_dataset)} samples | 验证集: {len(val_dataset)} samples")
    
    # 2. 数据加载器
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
    
    # 3. 初始化单任务模型
    model = SingleTaskMLP(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
        bn_eps=config.batch_norm_eps
    ).to(config.device)
    
    # 打印单任务模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"单任务模型参数: {total_params/1e6:.2f}M")
    
    # 4. 损失和优化器（单任务专用）
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 5. 训练过程
    best_val_loss = float('inf')
    scaler = torch.cuda.amp.GradScaler() if config.device.type == 'cuda' else None
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Task {task_var} | Epoch {epoch+1}/{config.epochs}")
        
        for rad_batch, ml_batch in pbar:
            rad_batch = rad_batch.to(config.device, non_blocking=True)
            ml_batch = ml_batch.to(config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
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
            
            batch_loss = loss.item() if not torch.isnan(loss) else 0.0
            train_loss += batch_loss * rad_batch.size(0)
            avg_loss = train_loss / ((pbar.n + 1) * config.batch_size)
            pbar.set_postfix({"Train Loss": f"{avg_loss:.6f}"})
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rad_batch, ml_batch in val_loader:
                rad_batch = rad_batch.to(config.device, non_blocking=True)
                ml_batch = ml_batch.to(config.device, non_blocking=True)
                
                outputs = model(rad_batch)
                loss = criterion(outputs, ml_batch)
                batch_loss = loss.item() if not torch.isnan(loss) else 0.0
                val_loss += batch_loss * rad_batch.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 保存最佳模型（标注中下层）
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"best_mlp_midlow80level_{task_var}.pth"  # 文件名标注中下层
            torch.save({
                'model_state_dict': model.state_dict(),
                'task_var': task_var,
                'level_range': '57~136 (250~1000hPa)',  # 记录层级范围
                'normalize_params': {
                    'ml_min': dataset.ml_min,
                    'ml_max': dataset.ml_max,
                    'rad_min': dataset.rad_min,
                    'rad_max': dataset.rad_max
                }
            }, model_path)
        
        # 打印日志
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val Loss: {best_val_loss:.6f}")
    
    print(f"\n任务 {task_var} 训练完成！最佳模型保存至: {model_path}")
    return model_path

# ===================== 6. 主执行流程（批量训练16个任务） =====================
if __name__ == "__main__":
    # 1. 加载中下层80层全量数据（核心修改后）
    print("加载大气中下层80层（250~1000hPa）大气参数数据...")
    ml_data, rad_data = load_80level_data(config.ml_data_path, config.rad_data_path)
    
    # 2. 按优先级训练16个任务
    trained_models = {}
    for task_var in config.train_task_order:
        model_path = train_single_task(task_var, ml_data, rad_data)
        trained_models[task_var] = model_path
    
    # 3. 打印训练完成汇总
    print("\n===== 所有任务训练完成 =====")
    for var, path in trained_models.items():
        print(f"{var}: {path}")
    
    # 示例：加载温度t的模型并预测
    t_model_path = trained_models['t']
    checkpoint = torch.load(t_model_path, map_location=config.device)
    t_model = SingleTaskMLP(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        hidden_dims=config.hidden_dims
    ).to(config.device)
    t_model.load_state_dict(checkpoint['model_state_dict'])
    t_model.eval()
    
    # 验证预测（中下层温度）
    test_dataset = ERA5SingleTaskDataset(ml_data, rad_data, 't')
    test_rad, test_t = test_dataset[0]
    test_rad = test_rad.unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        pred_t = t_model(test_rad)
        pred_t_denorm = test_dataset.inverse_normalize(pred_t.cpu().numpy())
        true_t_denorm = test_dataset.inverse_normalize(test_t.numpy())
    
    print(f"\n中下层温度t预测示例（80层）:")
    print(f"近地面层真实值: {true_t_denorm[-1]:.4f} K")  # 最后一层是近地面
    print(f"近地面层预测值: {pred_t_denorm[0,-1]:.4f} K")
    print(f"对流层中层真实值: {true_t_denorm[40]:.4f} K")  # 第40层是对流层中层
    print(f"对流层中层预测值: {pred_t_denorm[0,40]:.4f} K")