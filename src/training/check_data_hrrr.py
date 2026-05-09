import os
import numpy as np

def read_hrrr_binary(file_path):
    print(f"\n📊 读取HRRR二进制文件：{os.path.basename(file_path)}")
    print("="*80)
    
    # 核心配置：HRRR二进制文件的参数规范（参考气象模式二进制格式）
    # 若解析结果异常，需根据实际格式调整：参数顺序、数据类型、通道数
    param_specs = {
        "temperature": {"dtype": np.float32, "grid_size": (1377, 1377)},  # 温度（HRRR标准网格1377×1377）
        "relative_humidity": {"dtype": np.float32, "grid_size": (1377, 1377)},  # 相对湿度
        "u_wind": {"dtype": np.float32, "grid_size": (1377, 1377)},  # 东西风
        "v_wind": {"dtype": np.float32, "grid_size": (1377, 1377)},  # 南北风
        "pressure": {"dtype": np.float32, "grid_size": (1377, 1377)},  # 气压
        "surface_precip": {"dtype": np.float32, "grid_size": (1377, 1377)}  # 地面降水
    }
    
    # 计算每个参数的字节数
    param_bytes = {}
    total_expected_bytes = 0
    for param, spec in param_specs.items():
        dtype = spec["dtype"]
        grid_shape = spec["grid_size"]
        data_count = grid_shape[0] * grid_shape[1]  # 网格总数据量
        byte_size = np.dtype(dtype).itemsize * data_count  # 该参数总字节数
        param_bytes[param] = {"byte_size": byte_size, "grid_shape": grid_shape, "dtype": dtype}
        total_expected_bytes += byte_size
    
    try:
        # 读取二进制文件
        with open(file_path, 'rb') as f:
            data = f.read()
        actual_bytes = len(data)
        print(f"文件信息：")
        print(f"- 实际文件字节数: {actual_bytes}")
        print(f"- 预期参数总字节数: {total_expected_bytes}")
        
        if actual_bytes != total_expected_bytes:
            print(f"⚠️  字节数不匹配！可能参数配置错误，建议调整param_specs")
            print(f"  - 实际/预期: {actual_bytes}/{total_expected_bytes}")
        
        # 按顺序解析每个参数（二进制文件按参数顺序存储）
        offset = 0
        for param_name, spec in param_bytes.items():
            byte_size = spec["byte_size"]
            grid_shape = spec["grid_shape"]
            dtype = spec["dtype"]
            
            # 提取当前参数的二进制数据
            param_data_raw = data[offset:offset+byte_size]
            if len(param_data_raw) < byte_size:
                print(f"\n❌ {param_name} 数据不完整，跳过")
                offset += byte_size
                continue
            
            # 转换为numpy数组并重塑为网格格式
            param_data = np.frombuffer(param_data_raw, dtype=dtype).reshape(grid_shape)
            
            # 输出参数详情
            print(f"\n【参数名称】: {param_name}")
            print(f"  - 在文件中的位置: 字节偏移 {offset} - {offset+byte_size}")
            print(f"  - 数据类型: {dtype}")
            print(f"  - 网格维度: {grid_shape[0]} × {grid_shape[1]}")
            print(f"  - 数据总量: {param_data.size} 个数据点")
            print(f"  - 数据范围: 最小值{np.nanmin(param_data):.2f}, 最大值{np.nanmax(param_data):.2f}")
            print(f"  - 数据样例（前3×3网格）:\n{param_data[:3, :3]}")
            
            # 更新偏移量
            offset += byte_size
        
        print(f"\n✅ 解析完成（共解析 {len(param_specs)} 个参数）")
    except Exception as e:
        print(f"\n❌ 解析失败：{str(e)}")
        print("💡 建议：1. 调整param_specs中的参数顺序/数据类型/网格大小；2. 确认文件是否为连续存储的二进制格式")

def main():
    # 匹配HRRR二进制文件（排除.py脚本）
    hrrr_files = [f for f in os.listdir('.') if 'hrrr' in f.lower() and not f.endswith('.py')]
    if not hrrr_files:
        print("❌ 未找到HRRR二进制文件")
        return
    for file in hrrr_files:
        read_hrrr_binary(os.path.abspath(file))
        print("\n" + "-"*100 + "\n")

if __name__ == "__main__":
    main()

