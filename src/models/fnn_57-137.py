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

# ===================== 1. 物理约束配置（适配中下层） =====================
class PhysicsConfig:
    # 大气中下层垂直分层（80层分为3层，适配0~10km）
    LOWER_LAYER = 0  # 低层（近地面）：0-30层（850~1000hPa）
    LOWER_UPPER = 30
    MID_LAYER = 30   # 中层（对流层）：30-60层（500~850hPa）
    MID_UPPER = 60
    HIGH_LAYER = 60  # 中高层（对流层顶）：60-80层（250~500hPa）
    HIGH_UPPER = 80
    
    # 中下层关键参数物理值域约束（更贴合0~10km特征）
    PHYSICS_BOUNDS = {
        't': (200.0, 320.0),    # 中下层温度：200K-320K（比高层更暖）
        'q': (0.0, 0.04),       # 中下层比湿：0-0.04 kg/kg（水汽主要集中区）
        'o3': (1e-8, 1e-5),     # 中下层臭氧：1e-8-1e-5 kg/kg
        'z': (0.0, 10000.0),    # 中下层高度：0-10000 m（适配0~10km）
        'u': (-60.0, 60.0),     # 中下层纬向风：-60-60 m/s（比高层小）
        'v': (-60.0, 60.0),     # 中下层经向风：-60-60 m/s
        'lnsp': (9.0, 15.0),    # 中下层气压对数：9-15（对应100~1000hPa）
        'cc': (0.0, 1.0)        # 云量：0-1
    }
    
    # 中下层垂直梯度平滑约束（更严格，对流层垂直变化显著）
    GRADIENT_LAMBDA = 5e-3
    # 值域约束损失权重
    BOUND_LAMBDA = 1e-2

# ===================== 2. 基础配置 =====================
class Config:
    # 数据路径
    ml_data_path = "E:/ERA5-ml-0.25-all-20100201-0000.nc"       
    rad_data_path = "E:/ERA5-rad-0.25-20100201-0000.nocloud.airs324_aqua.crtm.nc"  

    # 中下层80层+16参数配置（57~136层，250~1000hPa）
    n_levels = 80
    input_dim = 324
    task_output_dim = n_levels
    hidden_dims = [1024, 512, 256]  # 适配中下层物理约束的轻量化结构
    dropout_rate = 0.1
    batch_norm_eps = 1e-5
    
    # 训练参数（适配中下层数据特征）
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 50
    batch_size = 32
    train_ratio = 0.8
    val_ratio = 0.2
    grad_clip_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 16个大气参数（按物理类型分组）
    task_vars = [
        'crwc', 'cswc', 'etadot', 'z', 't', 'q', 'w', 'vo',
        'lnsp', 'd', 'u', 'v', 'o3', 'clwc', 'ciwc', 'cc'
    ]
    # 中下层核心物理参数优先级（对流层关键参数）
    core_vars = ['t', 'q', 'u', 'v', 'z', 'cc']

config = Config()
phy_config = PhysicsConfig()

# ===================== 3. 物理约束数据集（适配中下层） =====================
class ERA5PhysicsDataset(Dataset):
    def __init__(self, ml_data, rad_data, task_var, normalize=True):
        self.normalize = normalize
        self.task_var = task_var
        self.var_idx = config.task_vars.index(task_var)
        self.n_levels = config.n_levels
        
        # 1. 裁剪为中下层80层数据（已在load函数中处理）
        ml_data = ml_data[:, :config.n_levels * len(config.task_vars)]
        self.task_ml_data = ml_data[:, self.var_idx::len(config.task_vars)]  # shape: (N, 80)
        
        # 2. 中下层物理约束预处理：按参数类型裁剪至物理值域
        if task_var in phy_config.PHYSICS_BOUNDS:
            min_val, max_val = phy_config.PHYSICS_BOUNDS[task_var]
            self.task_ml_data = np.clip(self.task_ml_data, min_val, max_val)
        
        # 3. 数值裁剪与NaN填充（适配中下层数据分布）
        rad_data = np.clip(rad_data, -1e4, 1e4)
        self.task_ml_data = np.clip(self.task_ml_data, -1e4, 1e4)
        rad_data = np.nan_to_num(rad_data, nan=0.0, posinf=1e4, neginf=-1e4)
        self.task_ml_data = np.nan_to_num(self.task_ml_data, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 4. 中下层物理感知归一化（修复核心：统一为数组格式）
        if normalize:
            # 亮温归一化（保持原有逻辑）
            self.rad_min = np.min(rad_data, axis=0)
            self.rad_max = np.max(rad_data, axis=0)
            equal_mask = self.rad_max == self.rad_min
            self.rad_max[equal_mask] = self.rad_min[equal_mask] + 1e-8
            rad_data = 2 * (rad_data - self.rad_min) / (self.rad_max - self.rad_min) - 1
            
            # 中下层物理约束的参数归一化（修复：统一为数组）
            if task_var in phy_config.PHYSICS_BOUNDS:
                # 物理值域转为数组（匹配中下层80层维度）
                self.ml_min = np.full(self.n_levels, phy_config.PHYSICS_BOUNDS[task_var][0])
                self.ml_max = np.full(self.n_levels, phy_config.PHYSICS_BOUNDS[task_var][1])
            else:
                # 数据极值：按中下层80层计算（shape: 80）
                self.ml_min = np.min(self.task_ml_data, axis=0)
                self.ml_max = np.max(self.task_ml_data, axis=0)
            
            # 修复：处理equal_mask的维度匹配
            equal_mask = self.ml_max == self.ml_min
            if isinstance(equal_mask, bool):
                equal_mask = np.full(self.n_levels, equal_mask)
            self.ml_max[equal_mask] = self.ml_min[equal_mask] + 1e-8
            
            # 归一化计算（维度匹配）
            self.task_ml_data = 2 * (self.task_ml_data - self.ml_min) / (self.ml_max - self.ml_min) - 1
        
        # 最终值域约束（中下层专属）
        self.task_ml_data = np.clip(self.task_ml_data, -1, 1)
        rad_data = np.clip(rad_data, -1, 1)
        
        self.rad_data = torch.FloatTensor(rad_data)
        self.task_ml_data = torch.FloatTensor(self.task_ml_data)

    def __len__(self):
        return len(self.rad_data)

    def __getitem__(self, idx):
        return self.rad_data[idx], self.task_ml_data[idx]
    
    def inverse_normalize(self, normalized_data):
        """反归一化并还原中下层物理值域"""
        if not self.normalize:
            return normalized_data
        # 确保输入是二维数组
        if len(normalized_data.shape) == 1:
            normalized_data = normalized_data.reshape(1, -1)
        # 反归一化
        denorm_data = (normalized_data + 1) * (self.ml_max - self.ml_min) / 2 + self.ml_min
        # 中下层物理值域二次约束
        if self.task_var in phy_config.PHYSICS_BOUNDS:
            min_val, max_val = phy_config.PHYSICS_BOUNDS[self.task_var]
            denorm_data = np.clip(denorm_data, min_val, max_val)
        return denorm_data

# ===================== 4. 中下层80层数据加载（核心修改） =====================
def load_80level_data():
    """加载大气中下层80层数据（57~136层，250~1000hPa）"""
    ml_ds = xr.open_dataset(config.ml_data_path)
    rad_ds = xr.open_dataset(config.rad_data_path)
    
    # ===== 核心修改：截取中下层80层 =====
    total_levels = ml_ds.dims["level"]  # ERA5原始137层
    start_level = total_levels - config.n_levels  # 137-80=57，从第57层开始取最后80层
    end_level = total_levels  # 到137层结束
    selected_level_indices = list(range(start_level, end_level))  # 57~136层（中下层）
    
    # 验证中下层层级对应的气压（关键！打印确认）
    all_pressures = ml_ds["level"].values
    selected_pressures = all_pressures[selected_level_indices]
    print("===== 大气中下层80层验证 =====")
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
            # 提取中下层当前层、当前参数的数据，并过滤无效样本
            var_data = ml_ds[var].isel(time=0, level=level).values.flatten()[rad_valid]
            level_data.append(var_data)
        ml_data_list.append(np.stack(level_data, axis=1))
    
    ml_data = np.concatenate(ml_data_list, axis=1)
    
    # 内存安全裁剪（适配中下层数据量）
    max_samples = 80000
    if len(ml_data) > max_samples:
        idx = np.random.choice(len(ml_data), max_samples, replace=False)
        ml_data = ml_data[idx]
        rad_data = rad_data[idx]
    
    print(f"\n中下层80层数据加载完成 - 总样本数: {len(ml_data)}")
    return ml_data, rad_data

# ===================== 5. 物理约束FNN模型（适配中下层） =====================
class PhysicsConstrainedFNN(nn.Module):
    def __init__(self, input_dim, output_dim, task_var, hidden_dims, dropout_rate=0.1, bn_eps=1e-5):
        super(PhysicsConstrainedFNN, self).__init__()
        self.task_var = task_var
        self.n_levels = config.n_levels
        self.output_dim = output_dim
        
        # 1. 输入层：中下层物理特征增强（适配对流层特征）
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0], eps=bn_eps),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 2. 隐藏层：中下层分层感知（适配对流层垂直分层）
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
        
        # 3. 输出层：中下层物理约束分支
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        # 中下层权重：增强近地面/对流层中层/对流层顶的物理感知
        self.layer_weights = nn.Parameter(torch.ones(3, output_dim))  # 低/中/高三层权重
        
        # 4. 中下层物理约束激活：适配对流层参数值域
        if task_var in phy_config.PHYSICS_BOUNDS:
            # 针对中下层有物理值域的参数，使用Sigmoid/Softplus约束
            if task_var == 'cc':  # 云量：0-1
                self.output_activation = nn.Sigmoid()
            elif task_var in ['q', 'o3']:  # 中下层微量气体：非负
                self.output_activation = nn.Softplus()
            else:
                self.output_activation = nn.Tanh()  # 其他参数：对称值域
        else:
            self.output_activation = nn.Tanh()
        
        # 初始化中下层分层权重（近地面权重更高）
        nn.init.constant_(self.layer_weights[0], 1.2)  # 近地面层权重增强
        nn.init.constant_(self.layer_weights[1], 1.0)  # 对流层中层
        nn.init.constant_(self.layer_weights[2], 0.8)  # 对流层顶层

    def forward(self, x):
        # 1. 中下层特征提取
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 2. 基础输出
        out = self.output_layer(x)
        
        # 3. 中下层分层物理加权（适配对流层垂直分层）
        lower_weight = self.layer_weights[0].unsqueeze(0)  # 近地面层权重
        mid_weight = self.layer_weights[1].unsqueeze(0)    # 对流层中层权重
        high_weight = self.layer_weights[2].unsqueeze(0)   # 对流层顶层权重
        
        # 按中下层分配权重
        out[:, :phy_config.LOWER_UPPER] *= lower_weight[:, :phy_config.LOWER_UPPER]
        out[:, phy_config.MID_LAYER:phy_config.MID_UPPER] *= mid_weight[:, phy_config.MID_LAYER:phy_config.MID_UPPER]
        out[:, phy_config.HIGH_LAYER:] *= high_weight[:, phy_config.HIGH_LAYER:]
        
        # 4. 中下层物理约束激活
        out = self.output_activation(out)
        
        return out
    
    def physics_loss(self, pred, target):
        """新增：中下层专属物理约束损失函数"""
        loss = 0.0
        
        # 1. 中下层垂直梯度平滑约束（对流层垂直变化更显著，约束更强）
        grad_pred = torch.abs(pred[:, 1:] - pred[:, :-1])  # 垂直梯度
        grad_loss = phy_config.GRADIENT_LAMBDA * torch.mean(grad_pred)
        loss += grad_loss
        
        # 2. 中下层值域约束损失（参数不超出对流层物理范围）
        if self.task_var in phy_config.PHYSICS_BOUNDS:
            min_val, max_val = phy_config.PHYSICS_BOUNDS[self.task_var]
            # 反归一化后的值域约束（适配中下层数组格式）
            pred_denorm = (pred + 1) * (max_val - min_val) / 2 + min_val
            # 超出中下层值域的惩罚
            lower_violation = torch.clamp(min_val - pred_denorm, min=0.0)
            upper_violation = torch.clamp(pred_denorm - max_val, min=0.0)
            bound_loss = phy_config.BOUND_LAMBDA * (torch.mean(lower_violation) + torch.mean(upper_violation))
            loss += bound_loss
        
        return loss

# ===================== 6. 物理约束训练函数（适配中下层） =====================
def train_physics_fnn(task_var, ml_data, rad_data):
    """带物理约束的FNN训练（中下层80层专属）"""
    # 1. 创建中下层物理约束数据集
    dataset = ERA5PhysicsDataset(ml_data, rad_data, task_var)
    total_samples = len(dataset)
    train_size = int(config.train_ratio * total_samples)
    val_size = total_samples - train_size
    
    torch.manual_seed(42)
    np.random.seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n===== 训练中下层物理约束FNN：{task_var} (80层) =====")
    print(f"训练集: {len(train_dataset)} samples | 验证集: {len(val_dataset)} samples")
    
    # 2. 数据加载器（适配中下层数据）
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
    
    # 3. 初始化中下层物理约束FNN
    model = PhysicsConstrainedFNN(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        task_var=task_var,
        hidden_dims=config.hidden_dims,
        dropout_rate=config.dropout_rate,
        bn_eps=config.batch_norm_eps
    ).to(config.device)
    
    # 4. 损失函数：MSE + 中下层物理约束损失
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    scaler = torch.cuda.amp.GradScaler() if config.device.type == 'cuda' else None
    
    # 5. 训练过程（适配中下层特征）
    best_val_loss = float('inf')
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
                    # 总损失 = MSE损失 + 中下层物理约束损失
                    loss_mse = mse_loss(outputs, ml_batch)
                    loss_physics = model.physics_loss(outputs, ml_batch)
                    loss = loss_mse + loss_physics
            else:
                outputs = model(rad_batch)
                loss_mse = mse_loss(outputs, ml_batch)
                loss_physics = model.physics_loss(outputs, ml_batch)
                loss = loss_mse + loss_physics
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
            
            batch_loss = loss.item() if not torch.isnan(loss) else 0.0
            train_loss += batch_loss * rad_batch.size(0)
            avg_loss = train_loss / ((pbar.n + 1) * config.batch_size)
            pbar.set_postfix({"Train Loss": f"{avg_loss:.6f}", 
                              "MSE": f"{loss_mse.item():.6f}",
                              "Physics": f"{loss_physics.item():.6f}"})
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rad_batch, ml_batch in val_loader:
                rad_batch = rad_batch.to(config.device, non_blocking=True)
                ml_batch = ml_batch.to(config.device, non_blocking=True)
                
                outputs = model(rad_batch)
                loss_mse = mse_loss(outputs, ml_batch)
                loss_physics = model.physics_loss(outputs, ml_batch)
                loss = loss_mse + loss_physics
                
                batch_loss = loss.item() if not torch.isnan(loss) else 0.0
                val_loss += batch_loss * rad_batch.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 保存最佳模型（标注中下层+物理约束）
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"best_physics_fnn_midlow80level_{task_var}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'task_var': task_var,
                'level_range': '57~136 (250~1000hPa)',  # 记录中下层层级范围
                'normalize_params': {
                    'ml_min': dataset.ml_min,
                    'ml_max': dataset.ml_max,
                    'rad_min': dataset.rad_min,
                    'rad_max': dataset.rad_max
                },
                'physics_bounds': phy_config.PHYSICS_BOUNDS.get(task_var, None)
            }, model_path)
        
        # 打印日志
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Best Val Loss: {best_val_loss:.6f}")
    
    print(f"\n中下层物理约束FNN {task_var} 训练完成！模型保存至: {model_path}")
    return model_path

# ===================== 7. 主执行流程（中下层专属） =====================
if __name__ == "__main__":
    # 1. 加载中下层80层数据（物理约束预处理）
    print("加载大气中下层80层（250~1000hPa）大气参数数据...")
    ml_data, rad_data = load_80level_data()
    
    # 2. 按中下层核心参数优先级训练16个任务
    trained_models = {}
    # 先训练中下层核心物理参数，再训练次要参数
    train_order = config.core_vars + [v for v in config.task_vars if v not in config.core_vars]
    
    for idx, var in enumerate(train_order):
        print(f"\n[{idx+1}/16] 训练中下层物理约束FNN：{var}")
        model_path = train_physics_fnn(var, ml_data, rad_data)
        trained_models[var] = model_path
    
    # 3. 打印训练汇总
    print("\n===== 所有中下层物理约束FNN训练完成 =====")
    for var, path in trained_models.items():
        print(f"{var}: {path}")
    
    # 示例：加载温度t的中下层物理约束模型
    t_model_path = trained_models['t']
    checkpoint = torch.load(t_model_path, map_location=config.device, weights_only=False)
    t_model = PhysicsConstrainedFNN(
        input_dim=config.input_dim,
        output_dim=config.task_output_dim,
        task_var='t',
        hidden_dims=config.hidden_dims
    ).to(config.device)
    t_model.load_state_dict(checkpoint['model_state_dict'])
    t_model.eval()
    
    # 验证中下层物理约束效果
    test_dataset = ERA5PhysicsDataset(ml_data, rad_data, 't')
    test_rad, test_t = test_dataset[0]
    test_rad = test_rad.unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        pred_t = t_model(test_rad)
        pred_t_denorm = test_dataset.inverse_normalize(pred_t.cpu().numpy())
        true_t_denorm = test_dataset.inverse_normalize(test_t.numpy())
    
    print(f"\n中下层温度t物理约束预测示例（80层）:")
    print(f"近地面层真实值: {true_t_denorm[-1]:.2f}K | 预测值: {pred_t_denorm[0,-1]:.2f}K")
    print(f"对流层中层真实值: {true_t_denorm[40]:.2f}K | 预测值: {pred_t_denorm[0,40]:.2f}K")
    print(f"中下层物理值域约束: {phy_config.PHYSICS_BOUNDS['t']} | 预测值是否在值域内: {phy_config.PHYSICS_BOUNDS['t'][0] <= pred_t_denorm[0,-1] <= phy_config.PHYSICS_BOUNDS['t'][1]}")