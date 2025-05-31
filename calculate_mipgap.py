import os
import numpy as np
import re

def extract_mipgap(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        match = re.search(r'MIPGap: ([\d.]+)', content)
        if match:
            return float(match.group(1))
    return None

def process_directory(base_dir):
    mipgaps = []
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'routes.txt':
                file_path = os.path.join(root, file)
                mipgap = extract_mipgap(file_path)
                if mipgap is not None:
                    mipgaps.append(mipgap)
    
    if mipgaps:
        avg = np.mean(mipgaps)
        std = np.std(mipgaps)
        print(f"总实例数: {len(mipgaps)}")
        print(f"MIPgap 平均值: {avg:.4f}")
        print(f"MIPgap 标准差: {std:.4f}")
        print(f"最小 MIPgap: {min(mipgaps):.4f}")
        print(f"最大 MIPgap: {max(mipgaps):.4f}")
    else:
        print("没有找到 MIPgap 数据")

if __name__ == "__main__":
    base_dir = "optimization_results/154022_全範圍_limit300s_時速60"
    process_directory(base_dir) 