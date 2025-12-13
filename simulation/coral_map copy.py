import numpy as np
import math
import os

from param import SONAR_RADIUS, SONAR_ANGLE, CAM_FOV
from param import true_positive_rate, false_positive_rate


class CoralMap:
    def __init__(self, map_file=None, map_size=(400, 400), cell_size=0.25):
        """
        初始化珊瑚地图。map_size: (rows, cols) unit: cells
        cell_size: 每个格子代表多少米
        """
        self.cell_size = cell_size  # 每个格子代表多少米
        
        if map_file and os.path.exists(map_file):
            print(f"Loading map from {map_file}...")
            self.map = np.load(map_file)  # ground truth map 
            # 0: Sand, 1: Rock, 2: Coral
        else:
            print("Generating random map...")
            # 0: Sand, 1: Rock, 2: Coral
            self.map = np.zeros(map_size, dtype=np.int8)
            # 随机生成岩石
            noise = np.random.rand(*map_size)
            self.map[noise > 0.85] = 1 
            # 随机生成珊瑚簇
            coral_spots = np.random.rand(*map_size)
            self.map[(coral_spots > 0.98) & (self.map == 1)] = 2
            
        self.rows, self.cols = self.map.shape
        self.width_meters = self.cols * self.cell_size
        self.height_meters = self.rows * self.cell_size

        # 状态矩阵
        # 0: 未知, 1: Sonar探测过(模糊), 2: Camera确认过
        self.status_mask = np.zeros_like(self.map, dtype=np.int8) 
        self.confirmed_count = 0
        self.total_corals = np.sum(self.map == 2)

    def world_to_grid(self, x, y):
        """将物理坐标转换为网格坐标"""
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return r, c

    def get_sensor_mask(self, robot_x, robot_y, robot_theta, max_dist, fov_angle=None, shape='sector'):
        """
        高效计算传感器覆盖的网格掩码。
        """
        # --- 修复开始 ---
        # 计算切片所需的“搜索半径”
        # 如果是正方形，我们需要更大的空间来容纳旋转后的角 (乘以 sqrt(2))
        if shape == 'square':
            search_radius = max_dist * 1.415 # √2 ≈ 1.414，稍微大一点点防边缘误差
        else:
            search_radius = max_dist

        # 1. 使用 search_radius 计算包围盒 (Bounding Box)
        # 这样即使正方形旋转，角也不会超出这个切片范围
        r_min, c_min = self.world_to_grid(robot_x - search_radius, robot_y - search_radius)
        r_max, c_max = self.world_to_grid(robot_x + search_radius, robot_y + search_radius)
        # --- 修复结束 ---
        
        # 限制在地图范围内
        r_min, r_max = max(0, r_min), min(self.rows, r_max)
        c_min, c_max = max(0, c_min), min(self.cols, c_max)
        
        if r_min >= r_max or c_min >= c_max:
            return None, (0,0,0,0)

        # 2. 生成局部网格坐标 (保持不变)
        grid_rows = np.arange(r_min, r_max)
        grid_cols = np.arange(c_min, c_max)
        grid_x, grid_y = np.meshgrid(grid_cols * self.cell_size + self.cell_size/2, 
                                     grid_rows * self.cell_size + self.cell_size/2)
        
        # 3. 转换到机器人Body Frame (保持不变)
        dx = grid_x - robot_x
        dy = grid_y - robot_y
        
        # 修正：确保使用正确的坐标系转换（之前的修复中提到过）
        # Body X forward, Body Y left
        # 这里不需要改动，只要 robot_theta 是正确的即可
        cos_t = math.cos(-robot_theta)
        sin_t = math.sin(-robot_theta)
        
        local_x = dx * cos_t - dy * sin_t
        local_y = dx * sin_t + dy * cos_t
        
        # 4. 计算形状掩码
        mask = np.zeros_like(local_x, dtype=bool)
        
        if shape == 'sector': 
            dist_sq = local_x**2 + local_y**2
            angle_mask = np.abs(np.arctan2(local_y, local_x)) <= (fov_angle / 2.0)
            mask = (dist_sq <= max_dist**2) & (local_x > 0) & angle_mask
            
        elif shape == 'square': 
            # 注意：这里的判断逻辑不变，依然判断点是否在正方形内
            # 但因为上面的 r_min/r_max 范围变大了，现在我们可以捕捉到旋转后的角了
            half_side = max_dist 
            mask = (np.abs(local_x) <= half_side) & (np.abs(local_y) <= half_side)
            
        return mask, (r_min, r_max, c_min, c_max)

    def update_sensors(self, robot_x, robot_y, robot_theta):
        """更新传感器读数并统计分数"""
        
        # ==========================================
        # 0. 重置声纳状态 (Transient State Reset)
        # ==========================================
        # 规则：如果在 t_i+1 时刻没看到，状态就要变回 0。
        # 最简单的实现方式：每帧开始先把所有 "1" (Sonar Detected) 擦除变成 "0" (Unknown)
        # "2" (Confirmed) 是永久的，保持不变
        self.status_mask[self.status_mask == 1] = 0

        # --- 1. Camera Update (High Priority, High Confidence) ---
        # 假设相机视野是一个 2m x 2m 的矩形 (半边长 1.0)
        cam_half_size = CAM_FOV / 2
        cam_mask, (r1, r2, c1, c2) = self.get_sensor_mask(
            robot_x, robot_y, robot_theta, max_dist=cam_half_size, shape='square'
        )
        
        if cam_mask is not None:
            # 获取局部切片
            local_status = self.status_mask[r1:r2, c1:c2]
            local_map = self.map[r1:r2, c1:c2] # ground truth
            
            # 找到视野内且未确认的珊瑚
            # 条件: 在mask内 AND 是珊瑚(2) AND 未被确认(!=2)
            new_corals = cam_mask & (local_map == 2) & (local_status != 2)
            count = np.sum(new_corals)
            
            if count > 0:
                self.confirmed_count += count
                # 标记为已确认 (2)
                self.status_mask[r1:r2, c1:c2][new_corals] = 2
                
            # 相机也会把普通地形标记为已探索 (视觉上)
            visible_ground = cam_mask & (local_status != 2)
            # 如果仅仅是探索了地形但不是珊瑚，可以设个特殊值，或者仅用于渲染
            # 这里简单起见，只要相机看过，status就设为 2 (完全已知)
            self.status_mask[r1:r2, c1:c2][visible_ground] = 2

        # --- 2. Sonar Update (Probabilistic) ---
        # 假设声纳半径 5m，FOV 60度
        sonar_radius = SONAR_RADIUS
        sonar_fov = math.radians(SONAR_ANGLE)
        sonar_mask, (r1, r2, c1, c2) = self.get_sensor_mask(
            robot_x, robot_y, robot_theta, max_dist=sonar_radius, fov_angle=sonar_fov, shape='sector'
        )
        
        if sonar_mask is not None:
            local_map = self.map[r1:r2, c1:c2]
            local_status = self.status_mask[r1:r2, c1:c2]

            # 1. 计算距离以应用概率
            grid_rows = np.arange(r1, r2)
            grid_cols = np.arange(c1, c2)
            gx, gy = np.meshgrid(grid_cols * self.cell_size + self.cell_size/2, 
                                 grid_rows * self.cell_size + self.cell_size/2)
            dists = np.sqrt((gx - robot_x)**2 + (gy - robot_y)**2)
            dists[~sonar_mask] = np.inf  # 不在声纳范围内的点设为无穷大
            # 归一化距离 (0.0 ~ 1.0)
            d_norm = np.clip(dists / sonar_radius, 0, 1)
            # 2. 定义概率模型 (参考你的公式描述)
            p_tp = true_positive_rate(d_norm)

            # False Positive Rate (FP): 距离越远噪声越大。近处0.01，远处0.1
            # 这意味着远处有5%的概率把石头看成珊瑚
            p_fp = false_positive_rate(d_norm)

            # 3. 判断检测逻辑 (Measurement Model)
            random_roll = np.random.rand(*dists.shape)
            # 情况 A: 真实存在珊瑚 (Grid=2) -> 使用 TP 概率
            hit_real = (local_map == 2) & (random_roll < p_tp)
            
            # 情况 B: 真实是石头 (Grid!=2) -> 使用 FP 概率
            hit_fake = (local_map == 1) & (random_roll < p_fp)


            # 成功检测的条件
            detection = sonar_mask & (hit_real | hit_fake)
            # detection = sonar_mask & (hit_fake)
            
            # 更新状态为 1 (Sonar Detected / Suspected)
            # 只有当它还不是2(已确认)时才更新
            update_mask = detection & (local_status != 2)
            self.status_mask[r1:r2, c1:c2][update_mask] = 1
