import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import time

class InfoTrajectoryPlanner:
    def __init__(self, map_size=10.0, resolution=0.5):
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.n_cells = self.grid_size * self.grid_size
        
        # 传感器参数
        self.fov_range = 3.0
        self.fov_angle = np.deg2rad(90)
        self.lambda_sensor = 0.4  # 传感器效率
        
        # 动力学参数
        self.v_max = 1.5
        self.omega_max = 1.0
        self.dt = 0.4  # 单步时间可以稍大一点，减少总步数，加快计算
        
        # 生成栅格
        x_grid = np.linspace(self.resolution/2, self.map_size-self.resolution/2, self.grid_size)
        y_grid = np.linspace(self.resolution/2, self.map_size-self.resolution/2, self.grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x_grid, y_grid)
        self.X_flat = self.X_grid.flatten()
        self.Y_flat = self.Y_grid.flatten()

    def _differentiable_fov(self, x, y, theta):
        """可微视场模型 (复用之前的逻辑)"""
        # gamma_dist = 2.0
        # gamma_angle = 4.0
        gamma_dist = 5.0
        gamma_angle = 50.0
        
        dist_sq = (self.X_flat - x)**2 + (self.Y_flat - y)**2 + 1e-6
        dist = ca.sqrt(dist_sq)
        factor_dist = 1.0 / (1.0 + ca.exp(gamma_dist * (dist - self.fov_range)))
        
        dir_x = ca.cos(theta)
        dir_y = ca.sin(theta)
        vec_x = self.X_flat - x
        vec_y = self.Y_flat - y
        dot_prod = vec_x * dir_x + vec_y * dir_y
        cos_angle = dot_prod / (dist + 1e-6)
        
        min_cos = np.cos(self.fov_angle / 2.0)
        factor_angle = 1.0 / (1.0 + ca.exp(gamma_angle * (min_cos - cos_angle)))
        
        return factor_dist * factor_angle

    def plan(self, start_state, goal_state, initial_map_entropy):
        """
        一次性规划全局轨迹
        """
        # 1. 动态估算所需的 Horizon (步数)
        dist_total = np.linalg.norm(goal_state[:2] - start_state[:2])
        # 假设以 80% 的最大速度直飞所需的时间
        estimated_time = dist_total / (self.v_max * 0.8)
        # 增加 50% 的 buffer 时间用于探索
        buffer_factor = 1.5
        N = int((estimated_time * buffer_factor) / self.dt)
        # 限制最小和最大步数
        N = max(20, min(N, 80)) 
        N=30
        
        print(f"Planning Trajectory with Horizon N={N}, dt={self.dt}s...")

        # 2. 构建优化问题
        opti = ca.Opti() # 使用 Opti 接口，写起来比底层 MX 更直观
        
        # 变量
        # 状态序列: [x, y, theta] * (N+1)
        X = opti.variable(3, N+1)
        # 控制序列: [vx, vy, omega] * N
        U = opti.variable(3, N)
        
        # 参数
        MapEntropy0 = opti.parameter(self.n_cells)
        
        # 初始化
        # 初始状态约束
        opti.subject_to(X[:, 0] == start_state)
        opti.set_value(MapEntropy0, initial_map_entropy.flatten())
        
        # 成本函数累加器
        obj = 0
        
        # 临时变量记录地图熵演化
        current_entropy = MapEntropy0
        
        # 权重设置
        w_info = 5.0     # 鼓励探索
        w_goal = 0     # 过程中的引导
        w_final = 100.0  # 必须到达终点
        w_smooth = 0.1   # 控制平滑
        w_jerk   = 5.0   # 加速度变化率

        for k in range(N):
            # --- 动力学约束 ---
            # 简单的欧拉积分
            opti.subject_to(X[0, k+1] == X[0, k] + U[0, k] * self.dt)
            opti.subject_to(X[1, k+1] == X[1, k] + U[1, k] * self.dt)
            # 角度不需要归一化，直接累积
            opti.subject_to(X[2, k+1] == X[2, k] + U[2, k] * self.dt)
            
            # --- 物理限制 ---
            # 速度限制 (vx, vy)
            opti.subject_to(opti.bounded(-self.v_max, U[0, k], self.v_max))
            opti.subject_to(opti.bounded(-self.v_max, U[1, k], self.v_max))
            opti.subject_to(opti.bounded(-self.omega_max, U[2, k], self.omega_max))
            
            # 地图边界限制
            opti.subject_to(opti.bounded(0.5, X[0, k+1], self.map_size-0.5))
            opti.subject_to(opti.bounded(0.5, X[1, k+1], self.map_size-0.5))
            
            # --- 信息增益计算 (Update Proxy) ---
            # 在 k+1 时刻的位姿进行观测
            alpha = self._differentiable_fov(X[0, k+1], X[1, k+1], X[2, k+1])
            
            # 模拟熵减
            next_entropy = current_entropy * (1.0 - self.lambda_sensor * alpha)
            
            # 收益 = 熵的减少量
            step_info_gain = ca.sum1(current_entropy) - ca.sum1(next_entropy)
            
            # 更新熵用于下一步
            current_entropy = next_entropy
            
            # --- 成本函数 ---
            # 1. 信息收益 (负号)
            obj -= w_info * step_info_gain
            
            # 2. 能量损耗
            # obj += w_smooth * ca.sumsqr(U[:, k])
            if k > 0:
                delta_u = U[:, k] - U[:, k-1]
                obj += w_jerk * ca.sumsqr(delta_u)
            
            # 3. 过程中的距离引导 (防止乱跑)
            dist_sq = (X[0, k+1] - goal_state[0])**2 + (X[1, k+1] - goal_state[1])**2
            obj += w_goal * dist_sq

        # --- 终端约束 ---
        # 强制最后一步必须非常接近终点 (Soft constraint 但权重很大)
        final_dist_sq = (X[0, N] - goal_state[0])**2 + (X[1, N] - goal_state[1])**2 + (X[2, N] - goal_state[2])**2
        obj += w_final * final_dist_sq
        
        # 最小化总成本
        opti.minimize(obj)
        
        # --- 求解 ---
        # 初始猜测 (直线轨迹)
        # 让求解器更容易收敛
        opti.set_initial(X[0, :], np.linspace(start_state[0], goal_state[0], N+1))
        opti.set_initial(X[1, :], np.linspace(start_state[1], goal_state[1], N+1))
        
        opts = {'ipopt.print_level': 5, 'ipopt.max_iter': 500, 'ipopt.tol': 1e-3}
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            
            # 提取轨迹
            traj_x = sol.value(X[0, :])
            traj_y = sol.value(X[1, :])
            traj_theta = sol.value(X[2, :])
            final_entropy_map = sol.value(current_entropy)
            
            return np.vstack((traj_x, traj_y, traj_theta)).T, final_entropy_map
            
        except Exception as e:
            print(f"Optimization Failed: {e}")
            # 如果失败，返回调试值 (Debug values)
            return opti.debug.value(X).T, opti.debug.value(current_entropy)

# --- 测试脚本 ---
if __name__ == "__main__":
    # 1. 初始化
    planner = InfoTrajectoryPlanner(map_size=10.0, resolution=0.5)
    
    start = np.array([1.0, 1.0, 0.0])
    goal = np.array([9.0, 9.0, 0.0])
    # init_entropy = np.ones(planner.n_cells) # 全黑地图
    init_entropy = np.where(planner.X_flat < (planner.map_size / 2.0)+2, 1.0, 0.0).astype(float)

    
    # 2. 规划
    t0 = time.time()
    trajectory, final_map_entropy = planner.plan(start, goal, init_entropy)
    print(f"Plan finished in {time.time() - t0:.2f}s")
    
    # 3. 可视化
    plt.figure(figsize=(10, 10))
    
    # 背景：预测的最终熵地图
    entropy_grid = final_map_entropy.reshape(planner.grid_size, planner.grid_size)
    plt.imshow(entropy_grid, extent=[0, 10, 0, 10], origin='lower', cmap='gray_r', vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(label='Predicted Remaining Entropy')
    
    # 轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=2, label='Planned Trajectory')
    
    # 起终点
    plt.scatter(start[0], start[1], c='g', s=100, label='Start')
    plt.scatter(goal[0], goal[1], c='r', s=100, label='Goal')
    
    # 绘制 FOV 采样 (每隔几步画一个)
    # step_size = max(1, len(trajectory)//15)
    step_size = 1
    for i in range(0, len(trajectory), step_size):
        x, y, theta = trajectory[i]
        degree = np.rad2deg(theta)
        wedge = Wedge((x, y), planner.fov_range, degree - 45, degree + 45, color='yellow', alpha=0.3)
        plt.gca().add_patch(wedge)
    
    plt.legend()
    plt.title("One-Shot Information-Aware Trajectory Optimization")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()