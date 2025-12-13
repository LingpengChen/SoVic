SONAR_RADIUS = 5
SONAR_ANGLE = 60
CAM_FOV = 2

START_X = 0
START_Y = 0

# True Positive Rate (TP): 距离越近越准。近处0.95，远处0.7
def true_positive_rate(d_norm):
    p_tp = 0.95 - 0.25 * d_norm
    return p_tp

# False Positive Rate (FP): 距离越远噪声越大。近处0.0，远处0.1
def false_positive_rate(d_norm):
    p_fp = 0.0 + 0.01 * d_norm
    return p_fp

GAME_FPS = 20

# manual control:
KEY_VELOCITY = 1.0  # m/s
KEY_ANGULAR_VELOCITY = 1.0  # rad/s

MAP_DIR = "./map/planning_maps/Area_2_map_1/Area_2_map_1_0.25m.npy"
MAP_SIZE = (400, 400)  # cells
CELL_SIZE = 0.25  # meters


# GLOBAL PLANNER 参数
from dataclasses import dataclass
@dataclass
class GlobalPlannerConfig:
    """
    集中管理规划器的所有超参数和常量
    """
    # 地图属性
    pixel_resolution: float = 0.25  # 每个像素代表的实际距离 (米/pixel)

    # 节点生成参数
    grid_interval_m: float = 5.0  # 采样间隔 (米)

    # 优化器缩放因子 (OR-Tools只接受整数)
    # 如果 path_cost_scale_factor = 100, 代表 1米 = 100 cost unit
    path_cost_scale_factor: int = 100    
    prize_scale_factor: int = 10000 

    # 路径规划约束
    path_budget_meters: float = 1000.0 # 最大路径预算 (米)

    # 起点坐标 (米)
    depot_start_coords_m: tuple = (0.0, 0.0) 

    # 求解器限制
    solver_time_limit_sec: int = 3

GLOBAL_PLANNER_CONFIG = GlobalPlannerConfig(
        pixel_resolution=CELL_SIZE,      
        grid_interval_m=SONAR_RADIUS,    
        path_budget_meters=1500.0,   
        
        # 优化参数
        path_cost_scale_factor=1*100,      # 1米 = 100 cost
        prize_scale_factor=1*10000,   
        depot_start_coords_m = (START_X, START_Y),
        solver_time_limit_sec=3     
    )