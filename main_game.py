# ==========================================
# 2. 地图与传感器逻辑 (核心高效算法)
# ==========================================

import pygame
import numpy as np
import math
import sys
import os
from utils.robot import OmnidirectionalRobot
from utils.input_controller import InputController   

from utils.param import SONAR_RADIUS, SONAR_ANGLE, CAM_FOV
from utils.param import START_X, START_Y
from utils.param import GAME_FPS, MAP_DIR
from utils.param import KEY_VELOCITY, KEY_ANGULAR_VELOCITY

from utils.coral_map import CoralMap

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
        self.controller = InputController(key_velocity=KEY_VELOCITY, key_angular_velocity=KEY_ANGULAR_VELOCITY)
        
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