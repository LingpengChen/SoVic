# ==========================================
# Main Loop with Split-Screen Dashboard
# ==========================================

import pygame
import numpy as np
import math
import sys
import os
from simulation.robot import OmnidirectionalRobot
from simulation.input_controller import InputController   

from param import SONAR_RADIUS, SONAR_ANGLE, CAM_FOV
from param import START_X, START_Y
from param import GAME_FPS 
from param import MAP_DIR, MAP_SIZE, CELL_SIZE
from param import KEY_VELOCITY, KEY_ANGULAR_VELOCITY
from param import GLOBAL_PLANNER_CONFIG

from simulation.coral_map import CoralMap
from planner.belief_map import BeliefMap
from planner.global_planner import GlobalPlanner

class Game:
    def __init__(self):
        pygame.init()
        
        # --- 1. 布局配置 ---
        self.SIDEBAR_W = 260
        self.VIEW_W = 700
        self.HEIGHT = 700
        self.WIDTH = self.SIDEBAR_W + self.VIEW_W * 2  # 左栏 + 中图 + 右图
        
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("AUV Simulation Dashboard")
        
        # 定义三个区域的矩形 (Rect)
        self.rect_sidebar = pygame.Rect(0, 0, self.SIDEBAR_W, self.HEIGHT)
        self.rect_center  = pygame.Rect(self.SIDEBAR_W, 0, self.VIEW_W, self.HEIGHT)
        self.rect_right   = pygame.Rect(self.SIDEBAR_W + self.VIEW_W, 0, self.VIEW_W, self.HEIGHT)
        
        # Minimap 设置 (在左侧栏上方)
        self.minimap_size = (240, 240) # 稍微留点边距
        self.minimap_rect = pygame.Rect(10, 10, *self.minimap_size)
        
        # 颜色定义
        self.COLOR_SAND = (194, 178, 128)    
        self.COLOR_ROCK = (80, 80, 80)    
        self.COLOR_CORAL_GT = (255, 105, 180)  # 粉色 (Pink) 的 RGB 值
        self.COLOR_CORAL_CONFIRMED = (255, 50, 50)   # Camera确认 (红色)
        self.COLOR_CORAL_SONAR = (255, 165, 0)      # Sonar TP (橙色 - 真实信号)
        self.COLOR_SONAR_FP = (255, 255, 100)        # Sonar FP (淡黄色 - 虚假噪声/误报)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)

        # 初始化对象
        self.map_data = CoralMap(map_file=MAP_DIR, map_size=MAP_SIZE, cell_size=CELL_SIZE) 
        
        # 机器人
        start_x = START_X
        start_y = START_Y 
        # start_x = START_X if START_X else self.map_data.width_meters / 2
        # start_y = START_Y if START_Y else self.map_data.height_meters / 2
        self.robot = OmnidirectionalRobot(x=start_x, y=start_y)
        self.controller = InputController(key_velocity=KEY_VELOCITY, key_angular_velocity=KEY_ANGULAR_VELOCITY)
        
        # Belief Map
        substrate_prior = self.map_data.get_substrate_map()
        self.map_belief = BeliefMap(
            map_size=MAP_SIZE,
            cell_size=CELL_SIZE,
            substrate_prior=substrate_prior
        )

        # Global Planner
        self.global_planner = GlobalPlanner(GLOBAL_PLANNER_CONFIG)
        self.global_planner.get_prior_map(substrate_prior)
        self.all_nodes, optimal_nodes_sequence_indices, dist_m, total_prize = self.global_planner.plan_path()
        self.optimal_nodes = self.all_nodes[optimal_nodes_sequence_indices]

    def world_to_view_pixel(self, wx, wy, view_rect):
        """
        将世界坐标转换为指定 View 区域内的屏幕坐标。
        视图永远以机器人为中心。
        """
        # View 中心点 (屏幕坐标)
        cx = view_rect.x + view_rect.width // 2
        cy = view_rect.y + view_rect.height // 2
        
        # 缩放比例 (Zoom Level)
        scale = 15 
        
        # 计算偏移
        sx = cx + (wx - self.robot.x) * scale
        sy = cy + (wy - self.robot.y) * scale
        return int(sx), int(sy), scale

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(GAME_FPS) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Update
            ctrl = self.controller.get_control_input()
            self.robot.update(dt, ctrl)
            observations = self.map_data.get_observations(self.robot.x, self.robot.y, self.robot.theta)
            self.map_belief.update_belief(observations)

            # Render
            self.draw()
            pygame.display.flip()
            
        pygame.quit()
        sys.exit()

    def draw(self):
        # 清空背景
        self.screen.fill((10, 15, 20))
        
        # 1. 绘制左侧边栏 (Sidebar: Minimap + Stats)
        self.draw_sidebar()
        
        # 2. 绘制中间视图 (Ground Truth)
        self.draw_viewport(self.rect_center, mode="simulation", title="Simulation (observed Map)")
        
        # 3. 绘制右侧视图 (Belief Map)
        self.draw_viewport(self.rect_right, mode="belief", title="Robot Belief Map")
        
        # 绘制分割线
        pygame.draw.line(self.screen, (100, 100, 100), (self.SIDEBAR_W, 0), (self.SIDEBAR_W, self.HEIGHT), 2)
        pygame.draw.line(self.screen, (100, 100, 100), (self.SIDEBAR_W + self.VIEW_W, 0), (self.SIDEBAR_W + self.VIEW_W, self.HEIGHT), 2)

    def draw_sidebar(self):
        """绘制左侧信息栏"""
        # --- A. Minimap (Top Left) ---
        # 1. 生成 Ground Truth 的缩略图
        # 颜色映射: 0(Sand)->Brown, 1(Rock)->Gray, 2(Coral)->Red
        gt_map = self.map_data.map.T # Shape: (cols, rows)
        
        # 构建 RGB 数组
        map_rgb = np.zeros((*gt_map.shape, 3), dtype=np.uint8)
        
        # Sand
        map_rgb[gt_map == 0] = self.COLOR_SAND
        # Rock
        map_rgb[gt_map == 1] = self.COLOR_ROCK
        # Coral
        map_rgb[gt_map == 2] = self.COLOR_CORAL_GT
        
        # 创建 Surface 并缩放
        surf = pygame.surfarray.make_surface(map_rgb)
        surf = pygame.transform.scale(surf, self.minimap_size)
        
        # 绘制 Minimap
        self.screen.blit(surf, self.minimap_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.minimap_rect, 2)
        
        # 绘制 Minimap 上的机器人点
        scale_x = self.minimap_size[0] / self.map_data.width_meters
        scale_y = self.minimap_size[1] / self.map_data.height_meters
        mini_rx = self.minimap_rect.x + int(self.robot.x * scale_x)
        mini_ry = self.minimap_rect.y + int(self.robot.y * scale_y)
        pygame.draw.circle(self.screen, (0, 255, 0), (mini_rx, mini_ry), 4)

        # --- B. Stats (Bottom Left) ---
        start_y = 300
        line_h = 30
        
        texts = [
            ("Global Stats", self.title_font, (255, 255, 0)),
            (f"Total Corals: {self.map_data.total_corals}", self.font, (200, 200, 200)),
            (f"Confirmed:    {self.map_data.confirmed_count}", self.font, (50, 255, 50)),
            # (f"Confirmed:    {self.map_belief.confirmation_map.sum()}", self.font, (50, 255, 50)),
            # 如果你有记录 GT 的 visit count，也可以加在这里
            ("Robot State", self.title_font, (0, 255, 255)),
            (f"X: {self.robot.x:.2f} m", self.font, (200, 200, 200)),
            (f"Y: {self.robot.y:.2f} m", self.font, (200, 200, 200)),
            (f"Head: {math.degrees(self.robot.theta):.1f}°", self.font, (200, 200, 200)),
            ("Controls", self.title_font, (150, 150, 150)),
            ("WASD: Move", self.font, (150, 150, 150)),
            ("Q / E: Rotate", self.font, (150, 150, 150))
        ]
        
        for i, (txt, font_obj, color) in enumerate(texts):
            s = font_obj.render(txt, True, color)
            self.screen.blit(s, (15, start_y + i * line_h))

    def draw_viewport(self, rect, mode, title):
        """
        通用视口绘制函数
        rect: 视口在屏幕上的矩形区域
        mode: 'simulation' or 'belief'
        """
        # 设置 Clip，防止画出界
        self.screen.set_clip(rect)
        
        # 计算视野范围 (Grid)
        scale = 15
        view_w_m = rect.width / scale
        view_h_m = rect.height / scale
        
        r_min, c_min = self.map_data.world_to_grid(self.robot.x - view_w_m/2 - 2, self.robot.y - view_h_m/2 - 2)
        r_max, c_max = self.map_data.world_to_grid(self.robot.x + view_w_m/2 + 2, self.robot.y + view_h_m/2 + 2)
        
        r_min = max(0, r_min); c_min = max(0, c_min)
        r_max = min(self.map_data.rows, r_max); c_max = min(self.map_data.cols, c_max)
        
        cell_px = int(self.map_data.cell_size * scale) + 1
        

        # --- 1. Draw Cells ---
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                wx = c * self.map_data.cell_size + self.map_data.cell_size/2
                wy = r * self.map_data.cell_size + self.map_data.cell_size/2
                sx, sy, _ = self.world_to_view_pixel(wx, wy, rect)
                
                # Check Visibility optimization (粗略裁剪)
                if not (rect.left - cell_px < sx < rect.right and rect.top - cell_px < sy < rect.bottom):
                    continue

                if mode == "simulation":
                    
                    # 渲染逻辑：
                    # 1. 如果相机看过 (status==2): 显示真实颜色 (Sand/Rock/Coral)
                    # 2. 如果声纳看过 (status==1): 
                    #    - 如果是Coral -> 显示橙色 (Detected)
                    #    - 否则 -> 显示模糊的岩石/沙地色 (这里简化处理，声纳只高亮目标)
                    # 3. 未知区域: 可以画暗一点或者不画
                    status = self.map_data.status_mask[r, c]
                    val = self.map_data.map[r, c]
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

                else:
                    # Belief 颜色逻辑 (热力图)
                 
                    prob = self.map_belief.probability_map[r, c]
                    intensity = int(prob * 255)
                    # intensity = int((math.exp(prob) - 1) / (math.e - 1) * 255)
                    color = (intensity, intensity, intensity)

                        
                pygame.draw.rect(self.screen, color, (sx - cell_px//2, sy - cell_px//2, cell_px, cell_px))

        # --- 1. DRAW PLANNING OVERLAY
        if mode == "simulation": 
            self.draw_planning_overlay(rect, scale)
        elif mode == "belief":
            # 如果你也想在 Belief 图上看到规划路径，也可以在这里调用
            # self.draw_planning_overlay(rect, scale)
            pass

        # --- 2. Draw Robot & Sensors Overlay ---
        self.draw_robot_overlay(rect)

        # --- 3. Draw Title Label ---
        label = self.title_font.render(title, True, (255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0, 150), (rect.x + 10, rect.y + 10, label.get_width()+10, 30))
        self.screen.blit(label, (rect.x + 15, rect.y + 15))
        
        # 解除 Clip
        self.screen.set_clip(None)

    def draw_robot_overlay(self, rect):
        """在指定的视口内绘制机器人和传感器范围"""
        sx, sy, scale = self.world_to_view_pixel(self.robot.x, self.robot.y, rect)
        
        # Sonar Arc
        sonar_r = SONAR_RADIUS * scale
        start_angle = self.robot.theta - math.radians(SONAR_ANGLE)/2
        end_angle = self.robot.theta + math.radians(SONAR_ANGLE)/2
        
        end_pos1 = (sx + sonar_r * math.cos(start_angle), sy + sonar_r * math.sin(start_angle))
        end_pos2 = (sx + sonar_r * math.cos(end_angle), sy + sonar_r * math.sin(end_angle))
        
        pygame.draw.line(self.screen, (0, 255, 0), (sx, sy), end_pos1, 1)
        pygame.draw.line(self.screen, (0, 255, 0), (sx, sy), end_pos2, 1)
        
        bbox = pygame.Rect(sx - sonar_r, sy - sonar_r, sonar_r*2, sonar_r*2)
        pygame.draw.arc(self.screen, (0, 255, 0), bbox, -end_angle, -start_angle, 1)
        
        # Camera Box
        cam_px = CAM_FOV * scale
        surf = pygame.Surface((cam_px, cam_px), pygame.SRCALPHA)
        pygame.draw.rect(surf, (255, 0, 0, 100), surf.get_rect(), 2)
        rot_surf = pygame.transform.rotate(surf, math.degrees(-self.robot.theta))
        rot_rect = rot_surf.get_rect(center=(sx, sy))
        self.screen.blit(rot_surf, rot_rect)
        
        # Robot Body
        r_rad = 6
        p1 = (sx + r_rad * math.cos(self.robot.theta), sy + r_rad * math.sin(self.robot.theta))
        p2 = (sx + r_rad * math.cos(self.robot.theta + 2.5), sy + r_rad * math.sin(self.robot.theta + 2.5))
        p3 = (sx + r_rad * math.cos(self.robot.theta - 2.5), sy + r_rad * math.sin(self.robot.theta - 2.5))
        pygame.draw.polygon(self.screen, (0, 255, 255), [p1, p2, p3])

    # self.optimal_nodes_m    self.all_nodes_m
    def draw_planning_overlay(self, rect, scale):
        """
        绘制路径规划相关的图层：网格线、候选节点、最优路径
        该函数应该在 draw_viewport 中被调用
        """
        # === 1. 绘制物理网格线 (Grid Lines) ===
        # 假设网格步长 (米)，你需要确保有这个变量，或者从 config 获取
        step_m = getattr(self, 'grid_interval_m', 5.0) 
        
        # 为了性能，只绘制视野范围内的网格线，而不是整个地图
        view_w_m = rect.width / scale
        view_h_m = rect.height / scale
        
        # 计算当前视野的物理边界
        min_x_m = self.robot.x - view_w_m / 2
        max_x_m = self.robot.x + view_w_m / 2
        min_y_m = self.robot.y - view_h_m / 2
        max_y_m = self.robot.y + view_h_m / 2

        # --- 画垂直线 (Vertical Lines) ---
        # 找到视野内第一条网格线的 x 坐标
        start_x_idx = int(min_x_m // step_m)
        end_x_idx = int(max_x_m // step_m) + 1
        
        for i in range(start_x_idx, end_x_idx + 1):
            x_m = i * step_m
            # 计算屏幕上的 x 坐标
            sx, _, _ = self.world_to_view_pixel(x_m, self.robot.y, rect)
            # 画线：从视口顶部到底部
            pygame.draw.line(self.screen, (255, 255, 255, 80), (sx, rect.top), (sx, rect.bottom), 1)

        # --- 画水平线 (Horizontal Lines) ---
        start_y_idx = int(min_y_m // step_m)
        end_y_idx = int(max_y_m // step_m) + 1
        
        for i in range(start_y_idx, end_y_idx + 1):
            y_m = i * step_m
            _, sy, _ = self.world_to_view_pixel(self.robot.x, y_m, rect)
            pygame.draw.line(self.screen, (255, 255, 255, 80), (rect.left, sy), (rect.right, sy), 1)

        # === 2. 绘制候选节点 (Candidate Nodes) ===
        # 假设 self.all_nodes_m 存储了所有候选节点坐标 [[x, y], ...]
        if self.all_nodes is not None:
            for node in self.all_nodes:
                nx, ny = node[0], node[1]
                
                # # 简单剔除视野外的点以提高性能
                # if not (min_x_m < nx < max_x_m and min_y_m < ny < max_y_m):
                #     continue
                    
                sx, sy, _ = self.world_to_view_pixel(nx, ny, rect)
                # 对应 plt.scatter(..., c='blue', s=10)
                # 绘制蓝色小圆点
                pygame.draw.circle(self.screen, (0, 0, 255), (int(sx), int(sy)), 3)

        # === 3. 绘制最优路径 (Best Path) ===
        # 假设 self.optimal_nodes 存储了规划好的路径点
        if self.optimal_nodes is not None:
            if len(self.optimal_nodes) > 1:
                # 将物理坐标转换为屏幕坐标列表
                screen_points = []
                for node in self.optimal_nodes:
                    sx, sy, _ = self.world_to_view_pixel(node[0], node[1], rect)
                    screen_points.append((sx, sy))
                
                # 绘制红线连线 (plt.plot(..., 'r-o'))
                pygame.draw.lines(self.screen, (255, 0, 0), False, screen_points, 2)
                
                # 绘制路径点的小圆点
                for sp in screen_points:
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(sp[0]), int(sp[1])), 3)

                # === 4. 标记起点 (Depot) ===
                # 起点是路径的第一个点，画个大一点的绿色点
                start_sx, start_sy = screen_points[0]
                pygame.draw.circle(self.screen, (0, 255, 0), (int(start_sx), int(start_sy)), 6)


if __name__ == "__main__":
    game = Game()
    game.run()