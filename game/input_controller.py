import pygame
import math

class InputController:
    def __init__(self, key_velocity=1.5, key_angular_velocity=1.0):
        self.key_velocity = key_velocity
        self.key_angular_velocity = key_angular_velocity
    
    def get_control_input(self):
        """处理键盘输入 并返回控制命令 [vx, vy, omega]"""
        keys = pygame.key.get_pressed()
        
        # 机体坐标系下的速度命令
        forward_velocity = 0.0
        side_velocity = 0.0
        angular_velocity = 0.0
        
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
        
        # 角速度控制 (Q/E)
        if keys[pygame.K_q]:
            angular_velocity = self.key_angular_velocity  # 逆时针旋转
        if keys[pygame.K_e]:
            angular_velocity = -self.key_angular_velocity  # 顺时针旋转
        
        # # 转换到世界坐标系
        # cos_theta = math.cos(robot_theta)
        # sin_theta = math.sin(robot_theta)
        
        # # 世界坐标系速度 = 旋转矩阵 * 机体坐标系速度
        # vx = forward_velocity * cos_theta - side_velocity * sin_theta
        # vy = forward_velocity * sin_theta + side_velocity * cos_theta
        
        vx = forward_velocity
        vy = side_velocity

        return [vx, vy, angular_velocity]
