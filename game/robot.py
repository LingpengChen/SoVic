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

        # 根据运动学模型更新位置和朝向
        self.x += (self.vx * math.cos(self.theta) - self.vy * math.sin(self.theta)) * dt
        self.y += (self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)) * dt
        self.theta += self.omega * dt

        # 角度归一化
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def get_state(self):
        return [self.x, self.y, self.theta]
