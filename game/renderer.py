import pygame
import math

class Renderer:
    def __init__(self, window_width=800, window_height=600, map_size=10.0):
        self.window_width = window_width
        self.window_height = window_height
        self.map_size = map_size
        
        # 显示参数
        self.map_offset_x = 50
        self.map_offset_y = 50
        self.map_pixel_size = 500
        self.pixels_per_meter = self.map_pixel_size / map_size
        
        # 字体
        self.font = pygame.font.Font(None, 24)
    
    def world_to_screen(self, world_x, world_y):
        """世界坐标转屏幕坐标"""
        screen_x = self.map_offset_x + world_x * self.pixels_per_meter
        screen_y = self.map_offset_y + (self.map_size - world_y) * self.pixels_per_meter  # Y轴翻转
        return int(screen_x), int(screen_y)
    
    def draw_information_map(self, screen, info_map):
        """绘制信息地图"""
        grid_size = info_map.grid_size
        cell_size = self.map_pixel_size / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算屏幕位置 - 翻转Y轴以匹配世界坐标系
                screen_x = self.map_offset_x + i * cell_size
                screen_y = self.map_offset_y + (grid_size - 1 - j) * cell_size
                
                # 根据熵值计算颜色 (黑色=未知, 白色=已知)
                entropy = info_map.entropy_map[i, j]
                gray_value = int((1.0 - entropy) * 255)
                color = (gray_value, gray_value, gray_value)
                
                # 绘制格子
                rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)  # 网格线
    
    def draw_robot(self, screen, robot, info_map):
        """绘制机器人"""
        screen_x, screen_y = self.world_to_screen(robot.x, robot.y)
        
        # 机器人本体 (圆形)
        robot_radius = int(0.1 * self.pixels_per_meter)
        pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), robot_radius)
        
        # 朝向箭头
        arrow_length = robot_radius * 1.5
        end_x = screen_x + arrow_length * math.cos(robot.theta)
        end_y = screen_y - arrow_length * math.sin(robot.theta)  # Y轴翻转
        pygame.draw.line(screen, (255, 255, 0), (screen_x, screen_y), (end_x, end_y), 3)
        
        # 视场范围
        self._draw_fov(screen, robot, info_map)
    
    def _draw_fov(self, screen, robot, info_map):
        """绘制视场范围"""
        screen_x, screen_y = self.world_to_screen(robot.x, robot.y)
        fov_radius = int(info_map.fov_range * self.pixels_per_meter)
        
        # 计算扇形的起始和结束角度
        half_fov = info_map.fov_angle / 2.0
        start_angle = robot.theta - half_fov
        end_angle = robot.theta + half_fov
        
        # 绘制扇形边界线
        start_x = screen_x + fov_radius * math.cos(start_angle)
        start_y = screen_y - fov_radius * math.sin(start_angle)
        end_x = screen_x + fov_radius * math.cos(end_angle)
        end_y = screen_y - fov_radius * math.sin(end_angle)
        
        pygame.draw.line(screen, (255, 255, 0), (screen_x, screen_y), (start_x, start_y), 2)
        pygame.draw.line(screen, (255, 255, 0), (screen_x, screen_y), (end_x, end_y), 2)
        pygame.draw.circle(screen, (255, 255, 0), (screen_x, screen_y), fov_radius, 1)
    
    def draw_ui(self, screen, robot, info_map):
        """绘制用户界面信息"""
        # 控制说明
        info_lines = [
            "Controls:",
            "W/S - Forward/Backward",
            "A/D - Left/Right Strafe",
            "Q/E - Rotate Left/Right",
            f"Position: ({robot.x:.1f}, {robot.y:.1f})",
            f"Orientation: {math.degrees(robot.theta):.1f}°",
            f"Total Entropy: {info_map.get_total_entropy():.1f}"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            screen.blit(text_surface, (self.window_width - 200, 50 + i * 25))
        
        # 地图边框
        map_rect = pygame.Rect(self.map_offset_x, self.map_offset_y, self.map_pixel_size, self.map_pixel_size)
        pygame.draw.rect(screen, (255, 255, 255), map_rect, 2)
