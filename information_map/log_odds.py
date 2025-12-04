import numpy as np
import matplotlib.pyplot as plt

def binary_entropy(p):
    # Clip p to avoid log(0)
    p = np.clip(p, 1e-6, 1-1e-6)
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_simulation(steps, alpha_seq):
    """
    对比两种模型的熵减过程
    steps: 步数
    alpha_seq: 每一步的观测强度 (数组)
    """
    
    # --- Model 1: Entropy Decay (原模型) ---
    h_decay = [1.0] # 初始熵 1.0
    lambda_param = 0.5 # 衰减系数
    
    curr_h = 1.0
    for alpha in alpha_seq:
        # H_new = H_old * (1 - lambda * alpha)
        curr_h = curr_h * (1 - lambda_param * alpha)
        h_decay.append(curr_h)
        
    # --- Model 2: DCIM (新模型) ---
    h_dcim = [1.0]
    lambda_accum = 0.0 # 初始累积信息 (Log-odds = 0)
    eta_param = 0.8    # 信息增益率
    
    for alpha in alpha_seq:
        # 1. 累积信息 (Linear Accumulation)
        lambda_accum += eta_param * alpha
        
        # 2. 映射到虚拟概率 (Sigmoid Mapping)
        # 注意：我们模拟从 0.5 (未知) 到 1.0 (确信) 的过程
        # 当 lambda=0, sigmoid=0.5; lambda -> inf, sigmoid -> 1
        virtual_p = sigmoid(lambda_accum)
        
        # 3. 计算熵 (Non-linear Entropy)
        curr_h_dcim = binary_entropy(virtual_p)
        h_dcim.append(curr_h_dcim)
        
    return h_decay, h_dcim

def visualize_dcim():
    steps = 20
    # 模拟一个持续观测的过程，每次观测强度为 0.8
    alpha_constant = np.ones(steps) * 0.8
    
    h_decay, h_dcim = run_simulation(steps, alpha_constant)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(h_decay, 'b--o', label='Entropy Decay Model (Exponential)')
    plt.plot(h_dcim, 'r-s', label='DCIM (Sigmoidal Log-odds)')
    
    plt.title('Dynamics Comparison: Decay vs. DCIM', fontsize=14)
    plt.xlabel('Simulation Steps', fontsize=12)
    plt.ylabel('Entropy / Uncertainty', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加注解
    plt.annotate('Fast initial drop,\nthen long tail', 
                 xy=(5, h_decay[5]), xytext=(8, 0.6),
                 arrowprops=dict(facecolor='blue', arrowstyle='->'), color='blue')
                 
    plt.annotate('S-shape onset (convex),\nthen smooth saturation', 
                 xy=(5, h_dcim[5]), xytext=(1, 0.2),
                 arrowprops=dict(facecolor='red', arrowstyle='->'), color='red')
    
    plt.show()

if __name__ == "__main__":
    visualize_dcim()