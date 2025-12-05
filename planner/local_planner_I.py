import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.patches as patches
import matplotlib.transforms as transforms

class UnifiedPlanner:
    def __init__(self):
        # 机器人参数
        self.dt = 0.5
        self.horizon = 20  # 预测步数
        self.cam_L = 2.0   # 相机方形边长 2m
        self.alpha = 0.5   # Sigmoid 平滑因子 (越小越陡峭)
        
        # 权重
        self.w_confirm = 100.0  # 确认目标的权重
        self.w_smooth = 1.0     # 平滑权重
        
        # 动力学限制
        self.v_max = 1.0
        self.w_max = 1.0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x / self.alpha))

    def get_camera_visibility(self, robot_pos, target_pos):
        """
        计算单个目标在单个机器人位姿下的可见性 (0~1)
        使用了 Soft-Square 近似
        robot_pos: [x, y, theta]
        target_pos: [tx, ty]
        """
        rx, ry, rtheta = robot_pos
        tx, ty = target_pos
        
        # 1. 将目标转到机器人局部坐标系
        dx = tx - rx
        dy = ty - ry
        cos_t = np.cos(-rtheta)
        sin_t = np.sin(-rtheta)
        local_x = cos_t * dx - sin_t * dy
        local_y = sin_t * dx + cos_t * dy
        
        # 2. Soft-Box Check: 检查是否在 [-L/2, L/2] 范围内
        # 我们希望 local_x < L/2 且 local_x > -L/2
        # 等价于 L/2 - |local_x| > 0
        in_x = self.sigmoid(self.cam_L/2 - np.abs(local_x))
        in_y = self.sigmoid(self.cam_L/2 - np.abs(local_y))
        
        return in_x * in_y

    def objective(self, U_flat, current_state, targets):
        """
        优化目标函数
        U_flat: 展平的控制输入 [v0, w0, v1, w1, ...]
        """
        U = U_flat.reshape(-1, 2)
        T = U.shape[0]
        
        cost = 0.0
        
        # 1. 状态前向传播 (Rollout)
        states = [current_state]
        x, y, theta = current_state
        
        traj_visibilities = np.zeros((T, len(targets)))
        
        for k in range(T):
            v, w = U[k]
            
            # 简单的运动学模型
            theta = theta + w * self.dt
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            states.append([x, y, theta])
            
            # 计算这一步对所有目标的可见性
            for i, target in enumerate(targets):
                traj_visibilities[k, i] = self.get_camera_visibility([x, y, theta], target)
                
            # 平滑性惩罚 (控制量大小 + 变化率)
            cost += self.w_smooth * (v**2 + w**2)
            
        # 2. 计算累计覆盖概率 (Diminishing Returns)
        # P(seen) = 1 - Prod(1 - p_t)
        # 意思是：只要在轨迹的任何一点看到了，就算看到了。
        # 我们最大化 P(seen)，即最小化 -P(seen)
        
        p_not_seen_total = np.ones(len(targets))
        
        for k in range(T):
            # 假设每一帧的检测概率上限是 0.9 (模拟传感器不确定性)
            p_detection_k = 0.9 * traj_visibilities[k] 
            p_not_seen_total *= (1 - p_detection_k)
            
        p_seen_total = 1 - p_not_seen_total
        
        # 奖励项是负的 Cost
        cost -= self.w_confirm * np.sum(p_seen_total)
        
        return cost

    def plan(self, start_pose, targets):
        # 初始猜测：静止
        T = self.horizon
        U0 = np.zeros(T * 2) 
        # 给一点初始速度防止梯度消失
        U0[0::2] = 0.5 
        
        # 边界约束
        bounds = []
        for _ in range(T):
            bounds.append((0, self.v_max))    # v
            bounds.append((-self.w_max, self.w_max)) # w
            
        print("正在优化轨迹...")
        res = minimize(self.objective, U0, args=(start_pose, targets), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'disp': True, 'maxiter': 100})
        
        return res.x.reshape(-1, 2)

# --- 可视化与测试 ---
if __name__ == "__main__":
    planner = UnifiedPlanner()
    
    start_pose = np.array([0, 0, 0]) # x, y, yaw
    # 模拟一些声纳探测到的候选点 (Candidates)
    targets = np.array([
        [2, 1],
        [3, -0.5],
        [5, 1.5],
        [6, 0]
    ])
    
    # 规划
    U_opt = planner.plan(start_pose, targets)
    
    # 重建轨迹用于绘图
    path = [start_pose]
    curr = start_pose.copy()
    for u in U_opt:
        curr[2] += u[1] * planner.dt
        curr[0] += u[0] * np.cos(curr[2]) * planner.dt
        curr[1] += u[0] * np.sin(curr[2]) * planner.dt
        path.append(curr.copy())
    path = np.array(path)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(path[:, 0], path[:, 1], 'b.-', label='Optimized Trajectory')
    plt.plot(targets[:, 0], targets[:, 1], 'r*', markersize=15, label='Candidates')
    plt.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    
    # 获取当前的 Axes 对象
    ax = plt.gca()

    # 画出每一帧的 Camera FOV (示意)
    for i in range(0, len(path), 2): # 每两帧画一次
        x, y, th = path[i]
        L = planner.cam_L
        
        # 1. 创建一个位于原点 (0,0) 的矩形
        # 我们不在这里指定 angle，而是通过变换来处理
        rect = patches.Rectangle((-L/2, -L/2), L, L, 
                                 edgecolor='green', fill=False, alpha=0.3)
        
        # 2. 创建变换：先旋转 (Rotate)，再平移 (Translate)
        # 注意：rotate 接受的是弧度
        # ax.transData 确保变换是基于数据坐标系的
        t = transforms.Affine2D().rotate(th).translate(x, y) + ax.transData
        
        # 3. 将变换应用到矩形上
        rect.set_transform(t)
        
        # 4. 添加到图表中
        ax.add_patch(rect)
        
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title("Unified Optimization: Path covers targets smoothly")
    plt.show()