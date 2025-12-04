import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z, gamma):
    """
    逻辑 Sigmoid 函数 (对应图片1中的公式)
    sigma(z) = 1 / (1 + exp(-gamma * z))
    gamma 越大，函数越陡峭（越接近硬截断）。
    """
    # 使用 np.exp 的时候防止溢出，通常可以加一个 clip 或者由调用者保证范围
    # 这里为了简单直接实现公式
    return 1.0 / (1.0 + np.exp(-gamma * z))

def differentiable_fov(sensor_pos, sensor_heading, target_pos, 
                       R_max, beta, gamma_d=5.0, gamma_a=20.0):
    """
    计算可微视场的观测强度 alpha (对应公式 6, 7, 8)
    
    参数:
    sensor_pos: np.array [x, y] 传感器位置
    sensor_heading: float 传感器朝向 (弧度)
    target_pos: np.array [x, y] 目标点位置 (可以是单个点，也可以是网格点矩阵)
    R_max: float 最大探测距离
    beta: float 半视场角 (Half-FOV angle), 弧度
    gamma_d: float 距离的锐度参数
    gamma_a: float 角度的锐度参数
    
    返回:
    alpha: 观测强度 (0.0 到 1.0 之间)
    """
    
    # 1. 计算距离向量和距离 d_i(x)
    diff = target_pos - sensor_pos
    # 计算欧几里得距离 (支持向量化操作)
    dist = np.linalg.norm(diff, axis=-1)
    
    # 2. 计算 Range Factor F_dist (公式 6)
    # 只要 dist < R_max，(R_max - dist) > 0，Sigmoid 趋近于 1
    # 只要 dist > R_max，(R_max - dist) < 0，Sigmoid 趋近于 0
    F_dist = sigmoid(R_max - dist, gamma_d)
    
    # 3. 计算角度因子 F_ang (公式 7)
    # 计算目标向量的方向
    # 为了避免除以0，加一个极小值
    dir_vec = diff / (dist[..., None] + 1e-6)
    
    # 传感器朝向向量
    sensor_vec = np.array([np.cos(sensor_heading), np.sin(sensor_heading)])
    
    # 计算 cos(phi)，即目标向量与传感器朝向的点积
    # phi 是目标与传感器朝向的夹角
    cos_phi = np.dot(dir_vec, sensor_vec)
    
    # 如果夹角 phi < beta (在FOV内)，则 cos(phi) > cos(beta)
    # 此时 (cos_phi - cos(beta)) > 0，Sigmoid 趋近于 1
    F_ang = sigmoid(cos_phi - np.cos(beta), gamma_a)
    
    # 4. 综合观测强度 alpha (公式 8)
    alpha = F_dist * F_ang
    
    return alpha

# ==========================================
# 下面是可视化代码，帮助理解“可微”的含义
# ==========================================

def visualize_fov():
    # 设置传感器参数
    R_max = 5.0            # 最大距离
    beta = np.radians(30)   # 半视场角 30度 (总共60度)
    R_max_differentiable = 5.0 - 0.1            # 最大距离
    beta_differentiable = np.radians(30-2)   # 半视场角 30度 (总共60度)

    sensor_pos = np.array([0.0, 0.0])
    sensor_heading = np.radians(90) # 朝上
    
    # 创建一个网格空间来模拟周围环境
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-2, 8, 200)
    X, Y = np.meshgrid(x, y)
    grid_pos = np.stack([X, Y], axis=-1)
    
    # 计算网格中每个点的观测强度
    # 尝试修改 gamma_d 和 gamma_a 看看边缘锐度的变化
    intensity = differentiable_fov(sensor_pos, sensor_heading, grid_pos, 
                                   R_max_differentiable, beta_differentiable, gamma_d=5.0, gamma_a=50.0)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.title("Differentiable FOV Model (Heatmap)\nValues define gradients for optimization")
    
    # 画出热力图
    cp = plt.contourf(X, Y, intensity, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Observation Intensity (alpha)')
    
    # 画出传感器位置
    plt.plot(sensor_pos[0], sensor_pos[1], 'ro', label='Sensor')
    # 画出传感器的朝向箭头
    plt.arrow(sensor_pos[0], sensor_pos[1], 
              np.cos(sensor_heading), np.sin(sensor_heading), 
              head_width=0.3, color='red')
    
    # 画出理论上的硬边界 (仅作对比)
    # 距离圆弧
    theta = np.linspace(sensor_heading - beta, sensor_heading + beta, 50)
    arc_x = sensor_pos[0] + R_max * np.cos(theta)
    arc_y = sensor_pos[1] + R_max * np.sin(theta)
    plt.plot(arc_x, arc_y, 'w--', alpha=0.5, label='Hard Boundary')
    plt.plot([sensor_pos[0], arc_x[0]], [sensor_pos[1], arc_y[0]], 'w--', alpha=0.5)
    plt.plot([sensor_pos[0], arc_x[-1]], [sensor_pos[1], arc_y[-1]], 'w--', alpha=0.5)
    
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    visualize_fov()