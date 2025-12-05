import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """基础 Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-z))

def get_f_dist(dist_array, R_max, gamma_d):
    """
    一维距离因子 F_dist
    输入: 
        dist_array: 一组距离值 (x)
        R_max: 最大探测距离
        gamma_d: 距离锐度参数
    输出:
        对应的 F_dist 值 (0~1)
    """
    # 公式: sigma( gamma * (R_max - d) )
    # 当 d < R_max 时，(R_max - d) > 0 -> 输出接近 1
    return sigmoid(gamma_d * (R_max - dist_array))

def get_f_ang(angle_deg_array, beta_deg, gamma_a):
    """
    一维角度因子 F_ang
    输入:
        angle_deg_array: 一组角度值 (单位: 度)
        beta_deg: 半视场角 beta (单位: 度)
        gamma_a: 角度锐度参数
    输出:
        对应的 F_ang 值 (0~1)
    """
    # 将角度转换为弧度进行计算
    phi_rad = np.radians(angle_deg_array)
    beta_rad = np.radians(beta_deg)
    
    # 公式: sigma( gamma * (cos(phi) - cos(beta)) )
    # 这里的输入变量实际上是 "余弦差"，但为了直观，我们针对 "角度" 绘图
    cosine_diff = np.cos(phi_rad) - np.cos(beta_rad)
    return sigmoid(gamma_a * cosine_diff)

# ==========================================
# 可视化部分
# ==========================================

def plot_1d_functions():
    # 设定模拟参数
    R_max = 10.0        # 最大距离 10米
    beta_deg = 30.0     # 半视场角 30度
    
    # 准备 x 轴数据
    # 距离：从 0 到 15米 (超过 R_max 以观察下降趋势)
    d_values = np.linspace(0, 15, 300)
    # 角度：从 0 到 60度 (超过 beta 以观察下降趋势)
    a_values = np.linspace(0, 60, 300)
    
    # 设定不同的 gamma 值进行对比
    gamma_list = [1.0, 5.0, 20.0, 100.0]
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 绘制 F_dist (距离因子) ---
    for g in gamma_list:
        y_dist = get_f_dist(d_values, R_max, g)
        label_text = f'$\gamma_d={g}$' + (' (Soft)' if g==1 else ' (Hard)' if g==100 else '')
        ax1.plot(d_values, y_dist, label=label_text, linewidth=2)
        
    ax1.axvline(x=R_max, color='k', linestyle='--', alpha=0.5, label=f'R_max ({R_max}m)')
    ax1.set_title(r'Range Factor $F_{dist}(d)$')
    ax1.set_xlabel('Distance $d$ (m)')
    ax1.set_ylabel('Value (0 to 1)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- 绘制 F_ang (角度因子) ---
    for g in gamma_list:
        y_ang = get_f_ang(a_values, beta_deg, g)
        ax2.plot(a_values, y_ang, label=f'$\gamma_a={g}$', linewidth=2)
        
    ax2.axvline(x=beta_deg, color='k', linestyle='--', alpha=0.5, label=f'Beta ({beta_deg}°)')
    ax2.set_title(r'Angular Factor $F_{ang}(\phi)$')
    ax2.set_xlabel('Angle $\phi$ (degrees)')
    ax2.set_ylabel('Value (0 to 1)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_1d_functions()