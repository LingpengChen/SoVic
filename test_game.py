import pygame
import numpy as np
import math
import sys
import os

from utils.param import SONAR_RADIUS, SONAR_ANGLE, CAM_FOV
from utils.param import START_X, START_Y
from utils.param import true_positive_rate, false_positive_rate
from utils.param import GAME_FPS, MAP_DIR
from utils.param import KEY_VELOCITY, KEY_ANGULAR_VELOCITY
# ==========================================
# 1. 机器人与控制器 (保留你的原始代码)
# ==========================================

class OmnidirectionalRobot:
    def __init__(self, x=10.0, y=10.0, theta=0.0, max_velocity=3.0, max_angular_velocity=2.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
    
    def update(self, dt, control_input):
        vx_cmd = np.clip(control_input[0], -self.max_velocity, self.max_velocity)
        vy_cmd = np.clip(control_input[1], -self.max_velocity, self.max_velocity)
        omega_cmd = np.clip(control_input[2], -self.max_angular_velocity, self.max_angular_velocity)

        self.vx = vx_cmd
        self.vy = vy_cmd
        self.omega = omega_cmd

        # 运动学更新
        self.x += (self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)) * dt
        self.y += (self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)) * dt
        self.theta += self.omega * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def get_state(self):
        return np.array([self.x, self.y, self.theta])

class InputController:
    def __init__(self, key_velocity=KEY_VELOCITY, key_angular_velocity=KEY_ANGULAR_VELOCITY):
        self.key_velocity = key_velocity
        self.key_angular_velocity = key_angular_velocity
    
    def get_control_input(self):
        keys = pygame.key.get_pressed()
        forward_velocity = 0.0
        side_velocity = 0.0
        angular_velocity = 0.0
        
        if keys[pygame.K_w]: forward_velocity = self.key_velocity
        if keys[pygame.K_s]: forward_velocity = -self.key_velocity
        if keys[pygame.K_a]: side_velocity = self.key_velocity
        if keys[pygame.K_d]: side_velocity = -self.key_velocity
        if keys[pygame.K_q]: angular_velocity = -self.key_angular_velocity
        if keys[pygame.K_e]: angular_velocity = self.key_angular_velocity

        return [forward_velocity, side_velocity, angular_velocity]

# ==========================================
# 2. 地图与传感器逻辑 (核心高效算法)
# ==========================================

class CoralMap:
    def __init__(self, map_file=None, grid_size=(200, 200), cell_size=0.5):
        self.cell_size = cell_size  # 每个格子代表多少米
        
        if map_file and os.path.exists(map_file):
            print(f"Loading map from {map_file}...")
            self.grid = np.load(map_file)
        else:
            print("Generating random map...")
            # 0: Sand, 1: Rock, 2: Coral
            self.grid = np.zeros(grid_size, dtype=np.int8)
            # 随机生成岩石
            noise = np.random.rand(*grid_size)
            self.grid[noise > 0.85] = 1 
            # 随机生成珊瑚簇
            coral_spots = np.random.rand(*grid_size)
            self.grid[(coral_spots > 0.98) & (self.grid == 1)] = 2
            
        self.rows, self.cols = self.grid.shape
        self.width_meters = self.cols * self.cell_size
        self.height_meters = self.rows * self.cell_size

        # 状态矩阵
        # 0: 未知, 1: Sonar探测过(模糊), 2: Camera确认过
        self.status_mask = np.zeros_like(self.grid, dtype=np.int8) 
        self.confirmed_count = 0
        self.total_corals = np.sum(self.grid == 2)

    def world_to_grid(self, x, y):
        """将物理坐标转换为网格坐标"""
        c = int(x / self.cell_size)
        r = int(y / self.cell_size)
        return r, c

    def get_local_grid_mask(self, robot_x, robot_y, robot_theta, max_dist, fov_angle=None, shape='sector'):
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
        cam_mask, (r1, r2, c1, c2) = self.get_local_grid_mask(
            robot_x, robot_y, robot_theta, max_dist=cam_half_size, shape='square'
        )
        
        if cam_mask is not None:
            # 获取局部切片
            local_status = self.status_mask[r1:r2, c1:c2]
            local_grid = self.grid[r1:r2, c1:c2]
            
            # 找到视野内且未确认的珊瑚
            # 条件: 在mask内 AND 是珊瑚(2) AND 未被确认(!=2)
            new_corals = cam_mask & (local_grid == 2) & (local_status != 2)
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
        sonar_mask, (r1, r2, c1, c2) = self.get_local_grid_mask(
            robot_x, robot_y, robot_theta, max_dist=sonar_radius, fov_angle=sonar_fov, shape='sector'
        )
        
        if sonar_mask is not None:
            local_grid = self.grid[r1:r2, c1:c2]
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
            hit_real = (local_grid == 2) & (random_roll < p_tp)
            
            # 情况 B: 真实是石头 (Grid!=2) -> 使用 FP 概率
            hit_fake = (local_grid == 1) & (random_roll < p_fp)


            # 成功检测的条件
            detection = sonar_mask & (hit_real | hit_fake)
            # detection = sonar_mask & (hit_fake)
            
            # 更新状态为 1 (Sonar Detected / Suspected)
            # 只有当它还不是2(已确认)时才更新
            update_mask = detection & (local_status != 2)
            self.status_mask[r1:r2, c1:c2][update_mask] = 1

# ==========================================
# 3. 游戏主循环与渲染
# ==========================================

class Game:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Find Coral - AUV Simulation")
        
        # 颜色定义
        # 颜色定义
        self.COLOR_BG = (20, 30, 40)         
        self.COLOR_SAND = (194, 178, 128)    
        self.COLOR_ROCK = (100, 100, 100)    
        
        # 状态颜色
        self.COLOR_CORAL_CONFIRMED = (255, 50, 50)   # Camera确认 (红色)
        self.COLOR_CORAL_SONAR = (255, 165, 0)       # Sonar TP (橙色 - 真实信号)
        self.COLOR_SONAR_FP = (255, 255, 100)        # Sonar FP (淡黄色 - 虚假噪声/误报)
        
        self.COLOR_ROBOT = (0, 255, 255)
        self.COLOR_SONAR_VIEW = (0, 255, 0, 50) # 半透明绿色
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        
        # 初始化对象
        self.map_data = CoralMap(map_file=MAP_DIR ,grid_size=(300, 300), cell_size=0.3) # 90m x 90m map
        start_x = START_X
        start_y = START_Y
        # start_x = self.map_data.width_meters / 2
        # start_y = self.map_data.height_meters / 2
        self.robot = OmnidirectionalRobot(x=start_x, y=start_y)
        self.controller = InputController()
        
        # 视窗摄像机 (跟随机器人)
        self.camera_offset_x = 0
        self.camera_offset_y = 0

    def world_to_screen(self, wx, wy):
        """将世界坐标转换为屏幕坐标 (以机器人为中心)"""
        # 屏幕中心
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        # 计算相对于屏幕中心的偏移 (缩放比例 1 meter = 15 pixels)
        scale = 15 
        sx = cx + (wx - self.robot.x) * scale
        sy = cy + (wy - self.robot.y) * scale
        return int(sx), int(sy)

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(GAME_FPS) / 1000.0  # seconds 表示每一帧之间的时间间隔（以秒为单位）
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 1. Update Robot
            ctrl = self.controller.get_control_input()
            self.robot.update(dt, ctrl)
            
            # 2. Update Sensors
            self.map_data.update_sensors(self.robot.x, self.robot.y, self.robot.theta)
            
            # 3. Render
            self.draw()
            
            pygame.display.flip()
            
        pygame.quit()
        sys.exit()

    def draw(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- 绘制地图 (局部绘制以提高性能) ---
        # 计算屏幕可见的世界范围
        scale = 15
        view_w_m = self.WIDTH / scale
        view_h_m = self.HEIGHT / scale
        
        r_min, c_min = self.map_data.world_to_grid(self.robot.x - view_w_m/2 - 2, self.robot.y - view_h_m/2 - 2)
        r_max, c_max = self.map_data.world_to_grid(self.robot.x + view_w_m/2 + 2, self.robot.y + view_h_m/2 + 2)
        
        r_min = max(0, r_min)
        c_min = max(0, c_min)
        r_max = min(self.map_data.rows, r_max)
        c_max = min(self.map_data.cols, c_max)
        
        # 创建局部图像表面
        # 这里为了演示清晰，我们逐个画矩形(虽然不是最高效的渲染方式，但对于几百个格子足够快)
        # 真正的高性能渲染应该生成一张大 Surface 或者是利用 blit_array
        
        cell_px = int(self.map_data.cell_size * scale) + 1 # +1 避免缝隙
        
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                val = self.map_data.grid[r, c]
                status = self.map_data.status_mask[r, c]
                
                wx = c * self.map_data.cell_size + self.map_data.cell_size/2
                wy = r * self.map_data.cell_size + self.map_data.cell_size/2
                sx, sy = self.world_to_screen(wx, wy)
                
                color = self.COLOR_BG
                
                # 渲染逻辑：
                # 1. 如果相机看过 (status==2): 显示真实颜色 (Sand/Rock/Coral)
                # 2. 如果声纳看过 (status==1): 
                #    - 如果是Coral -> 显示橙色 (Detected)
                #    - 否则 -> 显示模糊的岩石/沙地色 (这里简化处理，声纳只高亮目标)
                # 3. 未知区域: 可以画暗一点或者不画
                
                if status == 2: # Confirmed
                    if val == 0: color = self.COLOR_SAND
                    elif val == 1: color = self.COLOR_ROCK
                    elif val == 2: color = self.COLOR_CORAL_CONFIRMED
                elif status == 1: # Sonar Detected
                    if val == 2: 
                        # True Positive (TP): 地图上确实是珊瑚，且被声纳抓到了
                        color = self.COLOR_CORAL_SONAR # 橙色
                    elif val == 1:
                        # False Positive (FP): 地图上是石头，但被声纳误报为目标
                        # 显示为“鬼影”颜色(淡黄)，以便调试观察噪声分布
                        color = self.COLOR_SONAR_FP 
                else: # Unexplored
                    # 迷雾中的岩石和沙地，或者完全黑色
                    if val == 1: color = (40, 40, 40) # 隐约可见障碍物
                    else: color = (30, 30, 35)
                
                pygame.draw.rect(self.screen, color, (sx - cell_px//2, sy - cell_px//2, cell_px, cell_px))

        # --- 绘制传感器 FOV (可视化辅助) ---
        # 绘制声纳扇形 (两条线)
        sonar_r = SONAR_RADIUS * scale
        angle_fov = math.radians(SONAR_ANGLE)
        start_angle = self.robot.theta - angle_fov/2
        end_angle = self.robot.theta + angle_fov/2
        
        robot_screen_pos = self.world_to_screen(self.robot.x, self.robot.y)
        
        # 左边界
        end_pos1 = (robot_screen_pos[0] + sonar_r * math.cos(start_angle),
                    robot_screen_pos[1] + sonar_r * math.sin(start_angle))
        # 右边界
        end_pos2 = (robot_screen_pos[0] + sonar_r * math.cos(end_angle),
                    robot_screen_pos[1] + sonar_r * math.sin(end_angle))
        
        pygame.draw.line(self.screen, (0, 255, 0), robot_screen_pos, end_pos1, 1)
        pygame.draw.line(self.screen, (0, 255, 0), robot_screen_pos, end_pos2, 1)
        # 画个弧线意思一下
        rect_rect = pygame.Rect(robot_screen_pos[0]-sonar_r, robot_screen_pos[1]-sonar_r, sonar_r*2, sonar_r*2)
        pygame.draw.arc(self.screen, (0, 255, 0), rect_rect, -end_angle, -start_angle, 1)

        # 绘制相机矩形
        cam_size_px = CAM_FOV * scale # 3m box
        # 需要旋转矩形
        # 这里简化，只画一个红色的框跟随机器人
        surf = pygame.Surface((cam_size_px, cam_size_px), pygame.SRCALPHA)
        pygame.draw.rect(surf, (255, 0, 0, 100), surf.get_rect(), 2) # 边框
        rotated_surf = pygame.transform.rotate(surf, math.degrees(-self.robot.theta))
        rect = rotated_surf.get_rect(center=robot_screen_pos)
        self.screen.blit(rotated_surf, rect)

        # --- 绘制机器人 ---
        # 简单的三角形
        robot_radius = 5
        angle = self.robot.theta
        pt1 = (robot_screen_pos[0] + robot_radius * math.cos(angle), robot_screen_pos[1] + robot_radius * math.sin(angle))
        pt2 = (robot_screen_pos[0] + robot_radius * math.cos(angle + 2.5), robot_screen_pos[1] + robot_radius * math.sin(angle + 2.5))
        pt3 = (robot_screen_pos[0] + robot_radius * math.cos(angle - 2.5), robot_screen_pos[1] + robot_radius * math.sin(angle - 2.5))
        pygame.draw.polygon(self.screen, self.COLOR_ROBOT, [pt1, pt2, pt3])

        # --- UI ---
        score_text = self.font.render(f"Confirmed Corals: {self.map_data.confirmed_count} / {self.map_data.total_corals}", True, (255, 255, 255))
        pos_text = self.font.render(f"Pos: ({self.robot.x:.1f}, {self.robot.y:.1f})", True, (200, 200, 200))
        control_text = self.font.render("WASD: Move, QE: Rotate", True, (150, 150, 150))
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(pos_text, (10, 40))
        self.screen.blit(control_text, (10, 70))

if __name__ == "__main__":
    game = Game()
    game.run()