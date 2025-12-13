import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import time

class CameraConfirmationPlanner:
    def __init__(self):
        # 传感器参数 (Camera)
        self.cam_L = 2.0        # 方形视场边长 2m
        self.alpha_soft = 0.2   # Soft-Square 的平滑因子 (越小越陡峭)
        self.eta_cam = 1.0      # 相机每帧的确认效率 (Accumulation rate)
        self.lambda_sat = 2.0   # 饱和系数: P = 1 - exp(-lambda * Accumulation)
                                # lambda越大，需要的观测时间越短就能确认
        
        # 动力学参数 (AUV)
        self.v_max = 2.0
        self.omega_max = 0.8
        self.dt = 0.5           # 控制步长
        
        # 边界约束
        self.map_bounds = [0, 10, 0, 10]

        self.N = 30  # 默认步数

    def _soft_square_fov(self, robot_pose, target_pos):
        """
        可微的相机视场模型 (Soft-Square)
        robot_pose: [x, y, theta] (CasADi variable)
        target_pos: [tx, ty] (Constant)
        return: visibility score [0, 1]
        """
        rx, ry, rtheta = robot_pose[0], robot_pose[1], robot_pose[2]
        tx, ty = target_pos[0], target_pos[1]
        
        # 1. 坐标变换：将目标转换到 Body Frame
        dx = tx - rx
        dy = ty - ry
        cos_t = ca.cos(rtheta)
        sin_t = ca.sin(rtheta)
        
        # Body frame coordinates (rotated)
        # 纵向距离 (longitudinal)
        d_lon = cos_t * dx + sin_t * dy 
        # 横向距离 (lateral)
        d_lat = -sin_t * dx + cos_t * dy
        
        # 2. Soft-Box Check using Sigmoid
        # Sigmoid function: sigma(z) = 1 / (1 + exp(-z))
        # 我们希望 |d| < L/2  =>  L/2 - |d| > 0
        
        half_L = self.cam_L / 2.0
        # 为了数值稳定性，abs用 sqrt(x^2 + epsilon) 近似
        eps = 1e-3
        abs_lon = ca.sqrt(d_lon**2 + eps)
        abs_lat = ca.sqrt(d_lat**2 + eps)
        
        # x轴方向的可见性
        vis_lon = 1.0 / (1.0 + ca.exp(-(half_L - abs_lon) / self.alpha_soft))
        # y轴方向的可见性
        vis_lat = 1.0 / (1.0 + ca.exp(-(half_L - abs_lat) / self.alpha_soft))
        
        return vis_lon * vis_lat
    
    def visualize_sensor_model(self):
        """
        可视化 Soft-Square 传感器模型的形状
        """
        # 1. 创建网格
        range_limit = self.cam_L * 1.5
        x = np.linspace(-range_limit, range_limit, 200)
        y = np.linspace(-range_limit, range_limit, 200)
        X, Y = np.meshgrid(x, y)
        
        # 2. 用 Numpy 复现 Soft-Square 逻辑 (与 CasADi 逻辑一致)
        # 假设机器人位于 (0,0)，朝向 theta=0
        # 此时 d_lon = x, d_lat = y
        
        half_L = self.cam_L / 2.0
        
        # Sigmoid: 1 / (1 + exp(-x))
        # Logit_x = (L/2 - |x|) / alpha
        logit_x = (half_L - np.abs(X)) / self.alpha_soft
        logit_y = (half_L - np.abs(Y)) / self.alpha_soft
        
        vis_x = 1.0 / (1.0 + np.exp(-logit_x))
        vis_y = 1.0 / (1.0 + np.exp(-logit_y))
        
        Z = vis_x * vis_y
        
        # 3. 绘图
        plt.figure(figsize=(6, 5))
        # 绘制热力图
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Visibility Intensity')
        
        # 绘制理论上的硬边界 (红色虚线框)
        rect = patches.Rectangle((-half_L, -half_L), self.cam_L, self.cam_L, 
                                 linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)
        
        plt.title(f"Soft-Square Sensor Model\n(L={self.cam_L}, alpha={self.alpha_soft})")
        plt.xlabel("Body Frame X (m)")
        plt.ylabel("Body Frame Y (m)")
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plan(self, start_pose, goal_pose, candidates):
        """
        规划轨迹以最大化确认收益
        start_pose: [x, y, theta]
        goal_pose: [x, y] (Local Goal)
        candidates: [[cx1, cy1], [cx2, cy2], ...]
        """
        n_candidates = len(candidates)
        
        # 1. 动态估算 Horizon
        dist_total = np.linalg.norm(goal_pose[:2] - start_pose[:2])
        # 给充足的时间去绕路看点
        estimated_time = dist_total / (self.v_max * 0.8) 
        # 至少保证有足够时间覆盖所有点 (假设每个点耗时3秒)
        # estimated_time = max(estimated_time, n_candidates * 3.0)

        self.N = int(estimated_time / self.dt)
        self.N = max(20, min(self.N, 60)) # 限制步数范围
        self.N = 60 # 固定步数调试用

        print(f"Planning with N={self.N}, Candidates={n_candidates}...")

        # 2. 构建优化问题
        opti = ca.Opti()
        
        # 变量
        X = opti.variable(3, self.N+1) # [x, y, theta]
        U = opti.variable(2, self.N)   # [v, omega]

        # 潜变量：每个 Candidate 的累计关注度 (Accumulated Attention)
        Lambda = opti.variable(n_candidates, self.N+1)

        # 初始化约束
        opti.subject_to(X[:, 0] == start_pose)
        opti.subject_to(Lambda[:, 0] == 0) # 初始关注度为0
        
        obj = 0
        
        # 权重
        w_confirm = 100.0  # 核心收益
        w_goal    = 5.0    # 引导去终点
        w_energy  = 0.0    # 节省能量
        w_smooth  = 10.0   # 平滑控制

        for k in range(self.N):
            # --- 动力学约束 (Unicycle) ---
            v = U[0, k]
            w = U[1, k]
            theta = X[2, k]
            
            opti.subject_to(X[0, k+1] == X[0, k] + v * ca.cos(theta) * self.dt)
            opti.subject_to(X[1, k+1] == X[1, k] + v * ca.sin(theta) * self.dt)
            opti.subject_to(X[2, k+1] == X[2, k] + w * self.dt)
            
            # --- 物理限制 ---
            opti.subject_to(opti.bounded(0, v, self.v_max)) # 假设不能倒车
            opti.subject_to(opti.bounded(-self.omega_max, w, self.omega_max))
            
            opti.subject_to(opti.bounded(self.map_bounds[0], X[0, k+1], self.map_bounds[1]))
            opti.subject_to(opti.bounded(self.map_bounds[2], X[1, k+1], self.map_bounds[3]))
            
            # --- 观测动力学 (Observation Dynamics) ---
            # 计算当前 pose 对每个 target 的瞬时可见性
            # 注意：这里我们使用了向量化操作或者循环
            step_visibility = []
            for j in range(n_candidates):
                vis = self._soft_square_fov(X[:, k], candidates[j])
                step_visibility.append(vis)
            
            step_visibility = ca.vertcat(*step_visibility) # shape: (n_cand, 1)
            
            # 更新累计关注度 Lambda
            # Lambda[k+1] = Lambda[k] + eta * visibility * dt
            opti.subject_to(Lambda[:, k+1] == Lambda[:, k] + self.eta_cam * step_visibility * self.dt)
            
            # --- 成本函数累加 ---
            # 1. 平滑/能量
            if k > 0:
                delta_u = U[:, k] - U[:, k-1]
                obj += w_smooth * ca.sumsqr(delta_u)
            obj += w_energy * ca.sumsqr(U[:, k])
            
            # 2. 过程中的Goal引导 (Optional, 主要是为了防止陷在局部最优)
            dist_to_goal = (X[0, k] - goal_pose[0])**2 + (X[1, k] - goal_pose[1])**2
            obj += 0.1 * dist_to_goal

        # --- 终端目标函数 ---
        # 1. 确认收益 (Saturation Reward)
        # Maximize sum( 1 - exp(-lambda * Lambda_final) )
        # Equivalent to Minimize sum( exp(...) )
        final_lambda = Lambda[:, self.N]
        confirmation_loss = ca.sum1(ca.exp(-self.lambda_sat * final_lambda))
        obj += w_confirm * confirmation_loss
        
        # 2. 必须到达终点 (Hard constraint or heavy penalty)
        # 这里用 heavy penalty 保证它是 Soft Constraint
        final_dist_sq = (X[0, self.N] - goal_pose[0])**2 + (X[1, self.N] - goal_pose[1])**2
        obj += w_goal * 10.0 * final_dist_sq
        
        # 求解
        opti.minimize(obj)
        
        # 初始猜测 (直线)
        opti.set_initial(X[0, :], np.linspace(start_pose[0], goal_pose[0], self.N+1))
        opti.set_initial(X[1, :], np.linspace(start_pose[1], goal_pose[1], self.N+1))

        opts = {'ipopt.print_level': 5, 'ipopt.max_iter': 200, 'ipopt.tol': 1e-4}
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            
            return {
                'x': sol.value(X[0, :]),
                'y': sol.value(X[1, :]),
                'theta': sol.value(X[2, :]),
                'lambda': sol.value(Lambda)
            }
        except Exception as e:
            print(f"Optimization Failed: {e}")
            return {
                'x': opti.debug.value(X[0, :]),
                'y': opti.debug.value(X[1, :]),
                'theta': opti.debug.value(X[2, :]),
                'lambda': opti.debug.value(Lambda)
            }

# --- 测试 ---
if __name__ == "__main__":
    planner = CameraConfirmationPlanner()
    # planner.visualize_sensor_model()

    start = np.array([1.0, 1.0, 0.0]) # 朝向很重要，初始朝向0(向右)
    goal = np.array([9.0, 9.0])
    
    # # 模拟几个散落在路上的候选点
    # candidates = np.array([
    #     [2.5, 3.0], # 偏离直线
    #     [5.0, 5.0], # 在直线上
    #     [6.5, 4.0], # 需要往回一点?
    #     [8.0, 7.5]
    # ])
    # 生成随机点
    np.random.seed(42)  # 固定随机种子以便复现
    candidates = np.random.uniform(low=[1, 1], high=[9, 9], size=(20, 2))

    t0 = time.time()
    res = planner.plan(start, goal, candidates)
    print(f"Time: {time.time()-t0:.3f}s")
    
    # --- 可视化 ---
    plt.figure(figsize=(10, 10))
    
    # 画轨迹
    plt.plot(res['x'], res['y'], 'b.-', label='Trajectory', zorder=2)
    
    # 画起点终点
    plt.scatter(start[0], start[1], c='g', s=100, marker='^', label='Start')
    plt.scatter(goal[0], goal[1], c='r', s=100, marker='*', label='Goal')
    
    # 画 Candidates 及其被覆盖的程度
    final_lambdas = res['lambda'][:, -1]
    confirmed_probs = 1.0 - np.exp(-planner.lambda_sat * final_lambdas)
    
    for j, cand in enumerate(candidates):
        # 颜色深浅代表确认程度
        prob = confirmed_probs[j]
        color = plt.cm.Reds(prob) # 越红越确认
        plt.scatter(cand[0], cand[1], c=[color], s=150, edgecolors='k', zorder=3)
        plt.text(cand[0]+0.2, cand[1], f"{prob:.2f}", fontsize=9)
        
    # 画 Camera Footprint (每隔几步)
    ax = plt.gca()
    step = 2
    for i in range(0, len(res['x']), step):
        x, y, th = res['x'][i], res['y'][i], res['theta'][i]
        
        # 构造旋转矩形
        L = planner.cam_L
        # Matplotlib 的 Rectangle 是左下角，这里我们要以中心旋转
        rect = patches.Rectangle((-L/2, -L/2), L, L, 
                                 edgecolor='blue', fill=False, alpha=0.15)
        
        t = transforms.Affine2D().rotate(th).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.legend()
    # 修改标题以显示 N 的值
    plt.title(f"Camera Confirmation Trajectory Optimization (CasADi)\nSteps (N) = {planner.N}")
    plt.show()