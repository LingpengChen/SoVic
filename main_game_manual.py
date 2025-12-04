import pygame
from game.robot import OmnidirectionalRobot
from information_map.information_map import InformationMap
from game.input_controller import InputController
from game.renderer import Renderer

class Game:
    def __init__(self, window_width=800, window_height=600):
        pygame.init()
        
        # 初始化游戏组件
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Simple Robot Game")
        
        self.robot = OmnidirectionalRobot(x=2.0, y=2.0, theta=0.0)
        self.info_map = InformationMap(map_size=10.0, resolution=0.5)
        self.input_controller = InputController()
        self.renderer = Renderer(window_width, window_height)
        
        # 时间控制
        self.clock = pygame.time.Clock()
        self.dt = 0.05
    
    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def update(self):
        """更新游戏状态"""
        # 获取控制输入
        control_input = self.input_controller.get_control_input()
        
        # 更新机器人
        self.robot.update(self.dt, control_input)
        
        # 边界检查
        self.robot.x = max(0.2, min(9.8, self.robot.x))
        self.robot.y = max(0.2, min(9.8, self.robot.y))
        
        # 更新信息地图
        self.info_map.update_from_observation(self.robot.x, self.robot.y, self.robot.theta)
    
    def render(self):
        """渲染游戏"""
        self.screen.fill((0, 0, 0))  # 黑色背景
        self.renderer.draw_information_map(self.screen, self.info_map)
        self.renderer.draw_robot(self.screen, self.robot, self.info_map)
        self.renderer.draw_ui(self.screen, self.robot, self.info_map)
        pygame.display.flip()
    
    def run(self):
        """主游戏循环"""
        running = True
        
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
