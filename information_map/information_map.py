import numpy as np
import math

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
                # 计算栅格中心坐标
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
