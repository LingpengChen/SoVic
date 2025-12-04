import pygame
import numpy as np
import math
import time
from game.robot import OmnidirectionalRobot
from information_map.information_map import InformationMap
from mpc_controller import InfoAwareMPC, compute_predicted_trajectory
from game.renderer import Renderer

class AutoNavigationGame:
    def __init__(self, window_width=800, window_height=600):
        pygame.init()
        
        # 初始化游戏组件
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Auto Navigation with MPC")
        
        # 机器人和环境
        self.robot = OmnidirectionalRobot(x=1.0, y=1.0, theta=0.0)
        self.info_map = InformationMap(map_size=10.0, resolution=0.5)
        
        # MPC控制器
        self.mpc = InfoAwareMPC(
            horizon=15, 
            dt=0.1, 
            map_size=self.info_map.map_size, 
            resolution=self.info_map.resolution
        )
        
        # 渲染器
        self.renderer = Renderer(window_width, window_height)
        
        # 目标点
        self.goal = [8.5, 8.5, 0.0]
        
        # 时间控制
        self.clock = pygame.time.Clock()
        self.dt = 0.05  # 仿真时间步长
        
        # MPC步进控制
        self.step_mode = True  # 步进模式
        self.mpc_ready_to_execute = False  # MPC计算完成，等待执行
        self.mpc_executing = False  # 正在执行MPC
        self.execution_steps = 0  # 当前执行的步数
        self.steps_per_mpc = int(0.2 / self.dt)  # 每个MPC周期的步数
        
        # 当前MPC计划
        self.current_mpc_plan = None
        self.predicted_trajectory = None
        
        # 轨迹记录
        self.trajectory_history = [self.robot.get_state()]
        
        # 字体
        self.font = pygame.font.Font(None, 20)
    
    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # 重置仿真
                    self.reset_simulation()
                elif event.key == pygame.K_SPACE:
                    # 空格键触发下一步MPC计算或执行
                    if not self.mpc_executing and not self.mpc_ready_to_execute:
                        self.compute_next_mpc()
                    elif self.mpc_ready_to_execute:
                        self.start_mpc_execution()
        return True
    
    def reset_simulation(self):
        """重置仿真"""
        self.robot = OmnidirectionalRobot(x=1.0, y=1.0, theta=0.0)
        self.info_map = InformationMap(map_size=10.0, resolution=0.5)
        self.mpc = InfoAwareMPC(
            horizon=15, 
            dt=0.2, 
            map_size=self.info_map.map_size, 
            resolution=self.info_map.resolution
        )
        self.trajectory_history = [self.robot.get_state()]
        self.current_mpc_plan = None
        self.predicted_trajectory = None
        self.mpc_ready_to_execute = False
        self.mpc_executing = False
        self.execution_steps = 0
    
    def compute_next_mpc(self):
        """计算下一次MPC"""
        print(f"计算MPC... 当前位置: ({self.robot.x:.2f}, {self.robot.y:.2f})")
        
        # 获取当前状态
        current_state = np.array(self.robot.get_state())
        goal_state = np.array(self.goal)
        
        # 求解MPC
        try:
            self.current_mpc_plan = self.mpc.solve(
                current_state, 
                self.info_map.entropy_map, 
                goal_state
            )
            
            # 计算预测轨迹
            self.predicted_trajectory = compute_predicted_trajectory(
                current_state, 
                self.current_mpc_plan, 
                self.mpc.dt
            )
            
            self.mpc_ready_to_execute = True
            print(f"MPC计算完成！预测轨迹长度: {len(self.predicted_trajectory)}")
            print("按空格键执行MPC计划...")
            
        except Exception as e:
            print(f"MPC求解失败: {e}")
            # 使用简单的目标导向控制
            self.current_mpc_plan = self.get_fallback_control()
            self.predicted_trajectory = compute_predicted_trajectory(
                current_state, 
                self.current_mpc_plan, 
                0.2
            )
            self.mpc_ready_to_execute = True
    
    def start_mpc_execution(self):
        """开始执行MPC计划"""
        if self.current_mpc_plan is not None:
            self.mpc_executing = True
            self.mpc_ready_to_execute = False
            self.execution_steps = 0
            print("开始执行MPC计划...")
    
    def get_fallback_control(self):
        """获取备用控制策略"""
        goal_dir = np.array(self.goal[:2]) - np.array([self.robot.x, self.robot.y])
        goal_dist = np.linalg.norm(goal_dir)
        
        if goal_dist > 0.1:
            goal_dir = goal_dir / goal_dist
            u_fallback = np.zeros((3, 10))
            u_fallback[0, :] = goal_dir[0] * 1.0  # vx
            u_fallback[1, :] = goal_dir[1] * 1.0  # vy
            u_fallback[2, :] = 0.0  # omega
            return u_fallback
        else:
            return np.zeros((3, 10))
    
    def update(self):
        """更新游戏状态"""
        # 只在执行模式下更新机器人
        if self.mpc_executing and self.current_mpc_plan is not None:
            # 计算当前应该使用的控制输入
            mpc_index = min(self.execution_steps // self.steps_per_mpc, self.current_mpc_plan.shape[1] - 1)
            control_input = self.current_mpc_plan[:, mpc_index]
            
            # 应用控制输入
            self.robot.update(self.dt, control_input)
            
            # 边界检查
            self.robot.x = max(0.2, min(9.8, self.robot.x))
            self.robot.y = max(0.2, min(9.8, self.robot.y))
            
            # 更新信息地图
            self.info_map.update_from_observation(self.robot.x, self.robot.y, self.robot.theta)
            
            # 记录轨迹
            self.trajectory_history.append(self.robot.get_state())
            
            # 更新步数计数器
            self.execution_steps += 1
            
            # 检查是否完成当前MPC周期
            if self.execution_steps >= self.steps_per_mpc:
                self.mpc_executing = False
                self.execution_steps = 0
                print("MPC执行完成！按空格键计算下一次MPC...")
    
    def draw_predicted_trajectory(self):
        """绘制MPC预测轨迹"""
        if self.predicted_trajectory is not None and len(self.predicted_trajectory) > 1:
            # 绘制预测轨迹线
            points = []
            for state in self.predicted_trajectory:
                screen_x, screen_y = self.renderer.world_to_screen(state[0], state[1])
                points.append((screen_x, screen_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, points, 3)
            
            # 绘制预测轨迹点和机器人朝向
            for i, state in enumerate(self.predicted_trajectory[1:]):  # 跳过起始点
                screen_x, screen_y = self.renderer.world_to_screen(state[0], state[1])
                color_intensity = max(50, 255 - i * 15)  # 远处的点颜色更淡
                color = (0, color_intensity, 0)
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), 6)
                
                # 绘制机器人在预测位置的朝向
                arrow_length = 15
                end_x = screen_x + arrow_length * math.cos(state[2])
                end_y = screen_y - arrow_length * math.sin(state[2])  # Y轴翻转
                pygame.draw.line(self.screen, color, (screen_x, screen_y), (end_x, end_y), 2)
    
    def draw_trajectory_history(self):
        """绘制历史轨迹"""
        if len(self.trajectory_history) > 1:
            points = []
            for state in self.trajectory_history:
                screen_x, screen_y = self.renderer.world_to_screen(state[0], state[1])
                points.append((screen_x, screen_y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)
    
    def draw_goal(self):
        """绘制目标点"""
        screen_x, screen_y = self.renderer.world_to_screen(self.goal[0], self.goal[1])
        pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 12)
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), 8)
    
    def draw_info(self):
        """绘制信息面板"""
        # 状态描述
        if not self.mpc_ready_to_execute and not self.mpc_executing:
            status = "等待MPC计算 - 按空格键"
        elif self.mpc_ready_to_execute:
            status = "MPC已计算 - 按空格键执行"
        elif self.mpc_executing:
            status = f"执行MPC中 ({self.execution_steps}/{self.steps_per_mpc})"
        else:
            status = "准备下一次MPC"
        
        info_lines = [
            "Step-by-Step MPC Navigation",
            f"Status: {status}",
            f"Position: ({self.robot.x:.2f}, {self.robot.y:.2f})",
            f"Orientation: {math.degrees(self.robot.theta):.1f}°",
            f"Goal: ({self.goal[0]}, {self.goal[1]})",
            f"Distance to Goal: {np.linalg.norm(np.array([self.robot.x, self.robot.y]) - np.array(self.goal[:2])):.2f}m",
            f"Total Entropy: {self.info_map.get_total_entropy():.1f}",
            "",
            "Controls:",
            "SPACE - Next MPC Step",
            "R - Reset simulation", 
            "ESC - Exit"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (self.screen.get_width() - 280, 20 + i * 22))
    
    def render(self):
        """渲染游戏"""
        self.screen.fill((0, 0, 0))  # 黑色背景
        
        # 绘制信息地图
        self.renderer.draw_information_map(self.screen, self.info_map)
        
        # 绘制历史轨迹
        self.draw_trajectory_history()
        
        # 绘制MPC预测轨迹
        self.draw_predicted_trajectory()
        
        # 绘制机器人
        self.renderer.draw_robot(self.screen, self.robot, self.info_map)
        
        # 绘制目标点
        self.draw_goal()
        
        # 绘制信息面板
        self.draw_info()
        
        pygame.display.flip()
    
    def is_goal_reached(self):
        """检查是否到达目标"""
        distance = np.linalg.norm(np.array([self.robot.x, self.robot.y]) - np.array(self.goal[:2]))
        return distance < 0.3
    
    def run(self):
        """主游戏循环"""
        running = True
        
        print("开始步进式MPC导航...")
        print(f"起始位置: ({self.robot.x:.2f}, {self.robot.y:.2f})")
        print(f"目标位置: ({self.goal[0]}, {self.goal[1]})")
        print("按空格键开始第一次MPC计算...")
        
        while running:
            running = self.handle_events()
            
            if not self.is_goal_reached():
                self.update()
            else:
                if not hasattr(self, 'goal_reached_printed'):
                    print("目标到达！")
                    self.goal_reached_printed = True
            
            self.render()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("仿真结束")

if __name__ == "__main__":
    game = AutoNavigationGame()
    game.run()
