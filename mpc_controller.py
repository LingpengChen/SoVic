import casadi as ca
import numpy as np

class InfoAwareMPC:
    def __init__(self, horizon=20, dt=0.2, map_size=10.0, resolution=0.5):
        self.N = horizon
        self.dt = dt
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.n_cells = self.grid_size * self.grid_size
        
        # Warm start初始猜测
        self.u_prev = np.zeros((3, self.N))
        
        # 传感器参数 (与InformationMap保持一致)
        self.fov_range = 3.0
        self.fov_angle = np.deg2rad(90)  # 90度FOV
        self.lambda_sensor = 0.3
        
        # 生成栅格坐标中心点
        x_grid = np.linspace(self.resolution/2, self.map_size-self.resolution/2, self.grid_size)
        y_grid = np.linspace(self.resolution/2, self.map_size-self.resolution/2, self.grid_size)
        self.X_grid, self.Y_grid = np.meshgrid(x_grid, y_grid)
        self.X_flat = self.X_grid.flatten()  # [N_cells]
        self.Y_flat = self.Y_grid.flatten()  # [N_cells]
        
        # 构建CasADi优化问题
        self._build_optimal_control_problem()
    
    def _differentiable_fov(self, x, y, theta):
        """
        构建可微的视场 (Proxy Model)
        输入: 机器人的符号变量 x, y, theta
        输出: 所有栅格的观测强度 alpha [N_cells] (符号表达式)
        """
        # 参数：Sigmoid的锐度
        gamma_dist = 2.0
        gamma_angle = 5.0
        
        # 1. 距离因子 (Soft Range)
        dist_sq = (self.X_flat - x)**2 + (self.Y_flat - y)**2 + 1e-6
        dist = ca.sqrt(dist_sq)
        factor_dist = 1.0 / (1.0 + ca.exp(gamma_dist * (dist - self.fov_range)))
        
        # 2. 角度因子 (Soft Angle)
        dir_x = ca.cos(theta)
        dir_y = ca.sin(theta)
        vec_x = self.X_flat - x
        vec_y = self.Y_flat - y
        dot_prod = vec_x * dir_x + vec_y * dir_y
        cos_angle = dot_prod / (dist + 1e-6)
        
        min_cos = np.cos(self.fov_angle / 2.0)
        factor_angle = 1.0 / (1.0 + ca.exp(gamma_angle * (min_cos - cos_angle)))
        
        # 综合强度
        alpha = factor_dist * factor_angle
        return alpha
    
    def _build_optimal_control_problem(self):
        # 状态变量: [x, y, theta]
        n_states = 3
        n_controls = 3  # [vx, vy, omega]
        
        # 优化变量
        U = ca.MX.sym('U', n_controls, self.N)  # 控制序列
        P = ca.MX.sym('P', n_states + self.n_cells + 3)  # 参数: [初始状态(3) + 当前地图熵(N_cells) + 目标点(3)]
        
        # 解析参数
        X0 = P[:3]
        CurrentMapEntropy = P[3:3+self.n_cells]
        Goal = P[-3:]
        
        # 成本函数初始化
        obj = 0
        
        # 约束向量
        g = []
        
        # 初始状态
        st = X0
        map_entropy = CurrentMapEntropy
        
        # 权重
        w_info = 2.0      # 信息增益权重
        w_ctrl = 0.1      # 控制平滑权重
        w_goal = 1.0      # 目标引导权重
        w_final = 50.0    # 终点约束权重
        
        for k in range(self.N):
            u = U[:, k]  # 当前步控制 [vx, vy, omega]
            
            # --- A. 动力学模型 (运动学积分) ---
            x_next = st[0] + u[0] * self.dt
            y_next = st[1] + u[1] * self.dt
            theta_next = st[2] + u[2] * self.dt
            
            # 角度归一化
            theta_next = ca.atan2(ca.sin(theta_next), ca.cos(theta_next))
            
            st_next = ca.vertcat(x_next, y_next, theta_next)
            
            # --- B. 可微地图更新 ---
            alpha = self._differentiable_fov(st_next[0], st_next[1], st_next[2])
            
            # 预测下一时刻的地图熵
            reduction_factor = ca.fmax(0.0, ca.fmin(1.0, self.lambda_sensor * alpha))
            map_entropy_next = ca.fmax(0.0, map_entropy * (1.0 - reduction_factor))
            
            # 计算信息增益
            info_gain = ca.sum1(map_entropy) - ca.sum1(map_entropy_next)
            
            # --- C. 成本函数累加 ---
            # 1. 信息成本 (最大化增益 -> 最小化负增益)
            obj -= w_info * info_gain
            
            # 2. 控制能量成本
            obj += w_ctrl * ca.sumsqr(u)
            
            # 3. 目标引导成本
            obj += w_goal * ca.sumsqr(st_next[:2] - Goal[:2])
            
            # 更新状态和地图用于下一步
            st = st_next
            map_entropy = map_entropy_next
            
            # --- D. 约束 ---
            g.append(st[0])
            g.append(st[1])  # 位置约束
        
        # 终点约束
        obj += w_final * ca.sumsqr(st[:2] - Goal[:2])
        
        # 定义优化器
        opt_variables = ca.reshape(U, -1, 1)
        nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}
        
        # Solver选项
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        # 约束边界
        self.lbg = [0.5] * (2 * self.N)  # 地图边界约束
        self.ubg = [self.map_size - 0.5] * (2 * self.N)
        
        # 控制输入约束
        self.lbx = [-2.0, -2.0, -1.5] * self.N
        self.ubx = [2.0, 2.0, 1.5] * self.N
    
    def solve(self, current_state, current_map_entropy, goal_state):
        """
        求解MPC优化问题
        """
        # 构造参数向量
        P_val = np.concatenate((current_state, current_map_entropy.flatten(), goal_state))
        
        # 智能初始猜测 (Warm Start)
        u0 = np.zeros((3, self.N))
        if hasattr(self, 'u_prev') and self.u_prev is not None:
            # Shift策略：使用上次的解，向前移动一步
            u0[:, :-1] = self.u_prev[:, 1:]
            u0[:, -1] = 0
        else:
            # 第一次求解，朝向目标的初始猜测
            goal_dir = goal_state[:2] - current_state[:2]
            goal_dist = np.linalg.norm(goal_dir)
            if goal_dist > 1e-3:
                goal_dir = goal_dir / goal_dist
                u0[0, :] = goal_dir[0] * 0.8  # vx
                u0[1, :] = goal_dir[1] * 0.8  # vy
                u0[2, :] = 0.0  # omega
        
        try:
            # 求解
            sol = self.solver(x0=u0.flatten(), p=P_val, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)
            
            # 提取最优控制
            u_opt = sol['x'].full().reshape(3, self.N)
            
            # 保存当前解用于下次warm start
            self.u_prev = u_opt.copy()
            
            return u_opt
            
        except Exception as e:
            print(f"MPC求解失败: {e}")
            # 返回简单的朝向目标的控制
            goal_dir = goal_state[:2] - current_state[:2]
            goal_dist = np.linalg.norm(goal_dir)
            if goal_dist > 1e-3:
                goal_dir = goal_dir / goal_dist
                u_fallback = np.zeros((3, self.N))
                u_fallback[0, :] = goal_dir[0] * 0.5
                u_fallback[1, :] = goal_dir[1] * 0.5
                return u_fallback
            else:
                return np.zeros((3, self.N))

def compute_predicted_trajectory(current_state, u_plan, dt):
    """
    根据当前状态和控制序列计算预测轨迹
    """
    trajectory = [current_state.copy()]
    state = current_state.copy()
    
    for k in range(u_plan.shape[1]):
        # 应用控制输入
        state[0] += u_plan[0, k] * dt  # vx
        state[1] += u_plan[1, k] * dt  # vy
        state[2] += u_plan[2, k] * dt  # omega (注意这里不是减号)
        
        # 角度归一化
        state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
        
        trajectory.append(state.copy())
    
    return np.array(trajectory)
