import pygame
import numpy as np
import math

class OmnidirectionalRobot:
    def __init__(self, x=1.0, y=1.0, theta=0.0, max_velocity=2.0, max_angular_velocity=1.5):
        self.x = x
        self.y = y
        self.theta = theta  # 朝向角度 (弧度)
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity
        
        # 当前速度
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
    
    def update(self, dt, control_input):
        """
        更新机器人状态
        control_input: [vx_cmd, vy_cmd, omega_cmd]
        """
        # 限制控制输入
        vx_cmd = np.clip(control_input[0], -self.max_velocity, self.max_velocity)
        vy_cmd = np.clip(control_input[1], -self.max_velocity, self.max_velocity)
        omega_cmd = np.clip(control_input[2], -self.max_angular_velocity, self.max_angular_velocity)
        
        # 简单的一阶动力学 (直接设置速度)
        self.vx = vx_cmd
        self.vy = vy_cmd
        self.omega = omega_cmd
        
        # 积分更新位置和朝向
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.theta += self.omega * dt
        
        # 角度归一化
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def get_state(self):
        return [self.x, self.y, self.theta]

class InformationMap:
    def __init__(self, map_size=10.0, resolution=0.5):
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        
        # 初始化信息地图 (1.0表示完全未知, 0.0表示已知)
        self.entropy_map = np.ones((self.grid_size, self.grid_size))
        
        # 传感器参数
        self.fov_range = 3.0
        self.fov_angle = math.radians(90)  # 90度视场
        self.sensor_lambda = 0.3
    
    def update_from_observation(self, robot_x, robot_y, robot_theta):
        """根据机器人观测更新信息地图"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 计算栅格中心坐标 - 保持与显示一致的映射
                grid_x = (i + 0.5) * self.resolution
                grid_y = (j + 0.5) * self.resolution
                
                # 检查是否在视场内
                if self._is_in_fov(robot_x, robot_y, robot_theta, grid_x, grid_y):
                    # 计算观测强度 (基于距离)
                    dist = math.sqrt((grid_x - robot_x)**2 + (grid_y - robot_y)**2)
                    alpha = max(0, 1.0 - dist / self.fov_range)
                    
                    # 更新熵值
                    self.entropy_map[i, j] = max(0.0, self.entropy_map[i, j] * (1.0 - self.sensor_lambda * alpha))
    
    def _is_in_fov(self, robot_x, robot_y, robot_theta, grid_x, grid_y):
        """检查栅格是否在机器人视场内"""
        # 距离检查
        dist = math.sqrt((grid_x - robot_x)**2 + (grid_y - robot_y)**2)
        if dist > self.fov_range:
            return False
        
        # 角度检查
        vec_x = grid_x - robot_x
        vec_y = grid_y - robot_y
        angle_to_grid = math.atan2(vec_y, vec_x)
        angle_diff = abs(angle_to_grid - robot_theta)
        
        # 处理角度周期性
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        return angle_diff <= self.fov_angle / 2.0
    
    def get_total_entropy(self):
        return np.sum(self.entropy_map)

class SimpleRobotGame:
    def __init__(self, window_width=800, window_height=600):
        pygame.init()
        
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Simple Robot Game")
        
        # 游戏对象
        self.robot = OmnidirectionalRobot(x=2.0, y=2.0, theta=0.0)
        self.info_map = InformationMap(map_size=10.0, resolution=0.5)
        
        # 控制参数
        self.control_input = [0.0, 0.0, 0.0]  # [vx, vy, omega]
        self.key_velocity = 1.5
        self.key_angular_velocity = 1.0
        
        # 显示参数
        self.map_offset_x = 50
        self.map_offset_y = 50
        self.map_pixel_size = 500  # 地图在屏幕上的像素大小
        self.pixels_per_meter = self.map_pixel_size / 10.0  # 像素/米
        
        # 时间
        self.clock = pygame.time.Clock()
        self.dt = 0.05
        
        # 字体
        self.font = pygame.font.Font(None, 24)
    
    def handle_input(self):
        """处理键盘输入 - 相对于机器人机体坐标系"""
        keys = pygame.key.get_pressed()
        
        # 重置控制输入
        self.control_input = [0.0, 0.0, 0.0]
        
        # 机体坐标系下的速度命令
        forward_velocity = 0.0
        side_velocity = 0.0
        
        # 前进后退 (W/S)
        if keys[pygame.K_w]:
            forward_velocity = self.key_velocity  # 前进
        if keys[pygame.K_s]:
            forward_velocity = -self.key_velocity  # 后退
        
        # 左右平移 (A/D) 
        if keys[pygame.K_a]:
            side_velocity = self.key_velocity  # 左平移
        if keys[pygame.K_d]:
            side_velocity = -self.key_velocity  # 右平移
        
        # 转换到世界坐标系
        cos_theta = math.cos(self.robot.theta)
        sin_theta = math.sin(self.robot.theta)
        
        # 世界坐标系速度 = 旋转矩阵 * 机体坐标系速度
        self.control_input[0] = forward_velocity * cos_theta - side_velocity * sin_theta  # vx
        self.control_input[1] = forward_velocity * sin_theta + side_velocity * cos_theta  # vy
        
        # 角速度控制 (Q/E)
        if keys[pygame.K_q]:
            self.control_input[2] = self.key_angular_velocity  # 逆时针旋转
        if keys[pygame.K_e]:
            self.control_input[2] = -self.key_angular_velocity  # 顺时针旋转
    
    def world_to_screen(self, world_x, world_y):
        """世界坐标转屏幕坐标"""
        screen_x = self.map_offset_x + world_x * self.pixels_per_meter
        screen_y = self.map_offset_y + (10.0 - world_y) * self.pixels_per_meter  # Y轴翻转
        return int(screen_x), int(screen_y)
    
    def draw_information_map(self):
        """绘制信息地图"""
        grid_size = self.info_map.grid_size
        cell_size = self.map_pixel_size / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算屏幕位置 - 翻转Y轴以匹配世界坐标系
                screen_x = self.map_offset_x + i * cell_size
                screen_y = self.map_offset_y + (grid_size - 1 - j) * cell_size
                
                # 根据熵值计算颜色 (黑色=未知, 白色=已知)
                entropy = self.info_map.entropy_map[i, j]
                gray_value = int((1.0 - entropy) * 255)
                color = (gray_value, gray_value, gray_value)
                
                # 绘制格子
                rect = pygame.Rect(screen_x, screen_y, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)  # 网格线
    
    def draw_robot(self):
        """绘制机器人"""
        screen_x, screen_y = self.world_to_screen(self.robot.x, self.robot.y)
        
        # 机器人本体 (圆形)
        robot_radius = int(0.3 * self.pixels_per_meter)
        pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), robot_radius)
        
        # 朝向箭头
        arrow_length = robot_radius * 1.5
        end_x = screen_x + arrow_length * math.cos(self.robot.theta)
        end_y = screen_y - arrow_length * math.sin(self.robot.theta)  # Y轴翻转
        pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), (end_x, end_y), 3)
        
        # 视场范围 (扇形轮廓)
        self.draw_fov()
    
    def draw_fov(self):
        """绘制视场范围"""
        screen_x, screen_y = self.world_to_screen(self.robot.x, self.robot.y)
        fov_radius = int(self.info_map.fov_range * self.pixels_per_meter)
        
        # 计算扇形的起始和结束角度
        half_fov = self.info_map.fov_angle / 2.0
        start_angle = self.robot.theta - half_fov
        end_angle = self.robot.theta + half_fov
        
        # 绘制扇形边界线
        start_x = screen_x + fov_radius * math.cos(start_angle)
        start_y = screen_y - fov_radius * math.sin(start_angle)
        end_x = screen_x + fov_radius * math.cos(end_angle)
        end_y = screen_y - fov_radius * math.sin(end_angle)
        
        pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), (start_x, start_y), 2)
        pygame.draw.line(self.screen, (255, 255, 0), (screen_x, screen_y), (end_x, end_y), 2)
        
        # 绘制扇形圆弧 (简化版本)
        pygame.draw.circle(self.screen, (255, 255, 0), (screen_x, screen_y), fov_radius, 1)
    
    def draw_ui(self):
        """绘制用户界面信息"""
        # 控制说明
        info_lines = [
            "Controls:",
            "W/S - Forward/Backward",
            "A/D - Left/Right Strafe",
            "Q/E - Rotate Left/Right",
            f"Position: ({self.robot.x:.1f}, {self.robot.y:.1f})",
            f"Orientation: {math.degrees(self.robot.theta):.1f}°",
            f"Total Entropy: {self.info_map.get_total_entropy():.1f}"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (self.window_width - 200, 50 + i * 25))
        
        # 地图边框
        map_rect = pygame.Rect(self.map_offset_x, self.map_offset_y, self.map_pixel_size, self.map_pixel_size)
        pygame.draw.rect(self.screen, (255, 255, 255), map_rect, 2)
    
    def run(self):
        """主游戏循环"""
        running = True
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 处理输入
            self.handle_input()
            
            # 更新机器人
            self.robot.update(self.dt, self.control_input)
            
            # 边界检查
            self.robot.x = max(0.2, min(9.8, self.robot.x))
            self.robot.y = max(0.2, min(9.8, self.robot.y))
            
            # 更新信息地图
            self.info_map.update_from_observation(self.robot.x, self.robot.y, self.robot.theta)
            
            # 绘制
            self.screen.fill((0, 0, 0))  # 黑色背景
            self.draw_information_map()
            self.draw_robot()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    game = SimpleRobotGame()
    game.run()
