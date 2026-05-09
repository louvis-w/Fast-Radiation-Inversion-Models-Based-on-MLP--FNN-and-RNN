import numpy as np
import scipy.optimize as opt
import xarray as xr
import matplotlib.pyplot as plt

# ====================== 1. 反演配置（优化参数） ======================
class IASIInversionConfig:
    def __init__(self):
        self.press_levels = np.logspace(5, 2, 30)  # 30层气压（1000hPa-100hPa）
        self.n_levels = len(self.press_levels)
        self.n_params = self.n_levels * 2  # T(30) + q(30)
        # 优化反演参数（降低先验权重，提升收敛性）
        self.prior_cov = np.eye(self.n_params) * 10.0  # 增大先验误差，降低先验权重
        self.obs_error = np.ones(500) * 0.5  # 500个IASI敏感通道
        self.obs_cov = np.diag(self.obs_error **2)
        self.max_iter = 50  # 增加最大迭代次数
        self.conv_threshold = 1e-6  # 降低收敛阈值
        # 优化敏感矩阵（更合理的温湿贡献）
        self.sens_matrix = np.random.rand(500, self.n_levels) * 0.5  # 增强敏感系数

# ====================== 2. 替代正演函数（优化模拟逻辑） ======================
def rtm_forward(x, config, geo_params):
    """
    基于经验敏感矩阵的正演模拟（修复亮温偏移）
    x: 反演参数 [T0-T29, q0-q29]
    返回：模拟亮温
    """
    t_profile = x[:config.n_levels]  # 温度廓线 (K)
    q_profile = x[config.n_levels:]  # 水汽廓线 (g/kg)
    
    # 优化正演逻辑：更合理的亮温计算
    t_contribution = np.dot(config.sens_matrix, t_profile - 250)  # 温度偏离250K的贡献
    q_contribution = np.dot(config.sens_matrix * 0.2, q_profile - 5)  # 水汽偏离5g/kg的贡献
    geo_correction = (geo_params['vza'] - 30) * 0.05  # 观测角偏离30°的修正
    
    tb_sim = 280 + t_contribution + q_contribution + geo_correction  # 基础亮温设为280K
    return tb_sim

# ====================== 3. 代价函数+反演主逻辑（优化收敛性） ======================
def invert_iasi(iasi_nat_path):
    config = IASIInversionConfig()
    
    # 1. 读取/模拟IASI观测数据
    try:
        ds = xr.open_dataset(iasi_nat_path)
        tb_obs = ds['brightness_temperature'].values[:500] if 'brightness_temperature' in ds else None
        geo_params = {
            'vza': ds['viewing_zenith_angle'].values[0] if 'viewing_zenith_angle' in ds else 30.0,
            'surface_temp': ds['surface_temperature'].values[0] if 'surface_temperature' in ds else 290.0
        }
        ds.close()
    except:
        print("提示：使用模拟观测数据运行（请将.nat转为NetCDF后替换）")
        # 优化模拟观测数据（更接近真实IASI亮温）
        tb_obs = np.random.normal(280, 5, 500)  # 均值280K，标准差5K
        geo_params = {'vza': 30.0, 'surface_temp': 290.0}
    
    # 2. 优化先验廓线（更合理的初始值）
    t_prior = np.linspace(220, 288, config.n_levels)  # 对流层温度先验
    q_prior = np.linspace(0.1, 15, config.n_levels)    # 水汽先验
    x_prior = np.concatenate([t_prior, q_prior])
    
    # 3. 最优估计代价函数（添加数值稳定性）
    def cost_func(x):
        tb_sim = rtm_forward(x, config, geo_params)
        # 观测项（亮温差）
        obs_res = tb_obs - tb_sim
        obs_cov_inv = np.linalg.inv(config.obs_cov + 1e-6 * np.eye(len(config.obs_cov)))
        obs_term = np.dot(obs_res.T, np.dot(obs_cov_inv, obs_res))
        # 先验项（与先验廓线的偏差）
        prior_res = x - x_prior
        prior_cov_inv = np.linalg.inv(config.prior_cov + 1e-6 * np.eye(len(config.prior_cov)))
        prior_term = np.dot(prior_res.T, np.dot(prior_cov_inv, prior_res))
        return obs_term + prior_term
    
    # 4. 约束优化求解（优化参数提升收敛性）
    bounds = []
    bounds += [(180, 320) for _ in range(config.n_levels)]  # 温度边界
    bounds += [(0.01, 30) for _ in range(config.n_levels)]   # 水汽边界（修正0.0→0.01）
    
    res = opt.minimize(
        cost_func, x0=x_prior, method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': config.max_iter, 
            'gtol': config.conv_threshold,
            'ftol': 1e-6,  # 增加函数值收敛阈值
            'disp': True   # 输出优化过程
        }
    )
    
    # 5. 提取反演结果
    return {
        'press_levels': config.press_levels,
        't_retrieved': res.x[:config.n_levels],
        'q_retrieved': res.x[config.n_levels:],
        'tb_obs': tb_obs,
        'tb_sim': rtm_forward(res.x, config, geo_params),
        'converged': res.success,
        'n_iter': res.nit
    }
# ====================== 4. 可视化（仅修复标注，保留初始波动） ======================
def plot_results(result):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 温度廓线图（保留波动+清晰标注）
    ax1.plot(result['t_retrieved'], result['press_levels'], 'r-', linewidth=2, label='反演温度廓线')
    ax1.plot(np.linspace(220, 288, 30), result['press_levels'], 'b--', linewidth=2, label='先验温度廓线')
    ax1.set_yscale('log'), ax1.invert_yaxis()
    ax1.set_xlabel('温度 (K)', fontsize=12), ax1.set_ylabel('气压 (hPa)', fontsize=12)
    ax1.set_title('大气温度垂直廓线（反演结果 vs 先验值）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11), ax1.grid(alpha=0.3)
    
    # 2. 水汽廓线图（保留波动+清晰标注）
    ax2.plot(result['q_retrieved'], result['press_levels'], 'g-', linewidth=2, label='反演水汽廓线')
    ax2.plot(np.linspace(0.1, 15, 30), result['press_levels'], '--', color='orange', linewidth=2, label='先验水汽廓线')
    ax2.set_yscale('log'), ax2.invert_yaxis()
    ax2.set_xlabel('水汽混合比 (g/kg)', fontsize=12), ax2.set_ylabel('气压 (hPa)', fontsize=12)
    ax2.set_title('大气水汽垂直廓线（反演结果 vs 先验值）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11), ax2.grid(alpha=0.3)
    
    # 3. 亮温对比图（保留波动+清晰标注）
    ax3.plot(result['tb_obs'], 'b.', alpha=0.6, label='IASI观测亮温', markersize=3)
    ax3.plot(result['tb_sim'], 'r-', linewidth=1, label='模拟亮温')
    ax3.set_xlabel('通道索引', fontsize=12), ax3.set_ylabel('亮温 (K)', fontsize=12)
    ax3.set_title('IASI亮温（观测 vs 模拟）', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11), ax3.grid(alpha=0.3)
    
    # 4. 亮温残差图（保留波动+清晰标注）
    residual = result['tb_obs'] - result['tb_sim']
    ax4.plot(residual, 'k-', linewidth=1, label='亮温残差（观测-模拟）')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='残差为0基准线')
    ax4.set_xlabel('通道索引', fontsize=12), ax4.set_ylabel('亮温残差 (K)', fontsize=12)
    ax4.set_title('IASI亮温残差分布', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11), ax4.grid(alpha=0.3)
    
    plt.tight_layout(), plt.show()

# ====================== 5. 主程序 ======================
if __name__ == '__main__':
    iasi_file = "E:/IASI_xxx_1C_M01_20251101183252Z_20251101201156Z_N_O_20251101192303Z.nat"
    inversion_result = invert_iasi(iasi_file)
    
    # 输出关键信息
    print("="*50)
    print(f"反演收敛状态: {inversion_result['converged']}")
    print(f"迭代次数: {inversion_result['n_iter']}")
    print(f"平均亮温残差: {np.mean(np.abs(inversion_result['tb_obs']-inversion_result['tb_sim'])):.2f} K")
    print(f"反演温度范围: {np.min(inversion_result['t_retrieved']):.1f} ~ {np.max(inversion_result['t_retrieved']):.1f} K")
    print(f"反演水汽范围: {np.min(inversion_result['q_retrieved']):.1f} ~ {np.max(inversion_result['q_retrieved']):.1f} g/kg")
    print("="*50)
    
    plot_results(inversion_result)