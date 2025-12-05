import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
from dataclasses import dataclass

# ==========================================
# 1. 配置管理 (Configuration)
# ==========================================
# if we 
@dataclass
class PlannerConfig:
    """
    集中管理规划器的所有超参数和常量
    """
    # 地图属性
    pixel_resolution: float = 0.25  # 每个像素代表的实际距离 (米/pixel)
    
    # 节点生成参数
    grid_interval_pixels: int = 20  # 采样间隔 (pixels)，对应 5米
    fov_size_pixels: int = 20       # 视场大小 (pixels)，对应 5米 x 5m
    
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

# ==========================================
# 2. 核心算法模块
# ==========================================

class GlobalPlanner:
    def __init__(self, config: PlannerConfig):
        self.cfg = config

    def load_map(self, filepath: str) -> np.ndarray:
        print(f"[-] 正在加载地图: {filepath}")
        try:
            game_map = np.load(filepath)
            return game_map
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到地图文件: {filepath}")

    def generate_nodes_and_prizes(self, game_map: np.ndarray):
        """
        [修改重点] 
        1. 计算FOV内兴趣区域的几何重心 (Centroid) 代替网格中心。
        2. 将坐标直接转换为物理单位 (米)。
        """
        print("[-] 正在生成候选节点 (计算重心 & 转换为米)...")
        nodes_m = []  # 存储米制坐标
        prizes = []
        
        map_h, map_w = game_map.shape
        fov_half = self.cfg.fov_size_pixels // 2
        interval = self.cfg.grid_interval_pixels
        fov_area = self.cfg.fov_size_pixels ** 2
        res = self.cfg.pixel_resolution

        # 网格化滑动窗口
        for cy in range(fov_half, map_h - fov_half + 1, interval):
            for cx in range(fov_half, map_w - fov_half + 1, interval):
                # 定义 FOV 边界 (Pixel)
                min_y, max_y = cy - fov_half, cy + fov_half
                min_x, max_x = cx - fov_half, cx + fov_half
                
                # 提取 FOV 切片
                fov_slice = game_map[min_y:max_y, min_x:max_x]
                
                # 1. 提取兴趣点 (1:岩石, 2:珊瑚)
                # mask 是一个布尔矩阵，True表示感兴趣
                mask = (fov_slice == 1) | (fov_slice == 2)
                interest_count = np.sum(mask)
                
                if interest_count > 0:
                    # 计算归一化奖励
                    prize = interest_count / fov_area
                    
                    # --- [修改核心] 计算重心并转为米 ---
                    
                    # np.nonzero 返回的是相对于 slice 的局部坐标 (y_rel, x_rel)
                    rel_y_indices, rel_x_indices = np.nonzero(mask)
                    
                    # 转换为全图的绝对像素坐标
                    abs_y_indices = min_y + rel_y_indices
                    abs_x_indices = min_x + rel_x_indices
                    
                    # 计算像素坐标重心 (Mean)
                    centroid_y_px = np.mean(abs_y_indices)
                    centroid_x_px = np.mean(abs_x_indices)
                    
                    # 转换为物理坐标 (米)
                    # 注意：通常图像坐标系原点在左上或左下，这里假设 (0,0) 对应物理 (0m, 0m)
                    center_x_m = centroid_x_px * res
                    center_y_m = centroid_y_px * res
                    
                    nodes_m.append((center_x_m, center_y_m))
                    prizes.append(prize)
        
        # 插入起点 (Depot)，确保它是 (0.0, 0.0) 米
        nodes_m.insert(0, self.cfg.depot_start_coords_m)
        prizes.insert(0, 0.0)

        print(f"    生成了 {len(nodes_m) - 1} 个有效候选节点 (单位: 米)。")
        return np.array(nodes_m), np.array(prizes)

    def calculate_cost_matrix(self, nodes_m: np.ndarray) -> np.ndarray:
        """
        计算成本矩阵。
        因为输入 nodes_m 已经是米了，所以计算出的距离也是米。
        直接乘以 scale_factor 即可。
        """
        print("[-] 正在计算成本矩阵 (单位: 米 -> Cost Units)...")
        # 计算欧氏距离 (结果单位: 米)
        dist_matrix_m = distance.cdist(nodes_m, nodes_m, 'euclidean')
        
        # 放大并取整
        # 例如: 距离 5.5米, scale=100 -> cost=550
        scaled_matrix = (dist_matrix_m * self.cfg.path_cost_scale_factor).astype(int)
        return scaled_matrix

    def solve_orienteering_problem(self, prizes: np.ndarray, cost_matrix: np.ndarray):
        """
        求解定向问题
        """
        print(f"[-] 正在求解 (限时 {self.cfg.solver_time_limit_sec}s)...")
        
        num_nodes = len(cost_matrix)
        depot_idx = 0 

        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, depot_idx)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return cost_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 添加节点奖励 (惩罚项)
        scaled_prizes = (prizes * self.cfg.prize_scale_factor).astype(int)

        
        # === [添加代码 START] 打印成本与奖励分析 ===
        print(f"\n    [?] 成本/奖励 权衡分析:")
        print(f"        > 移动成本 (Cost Term): 1米 = {self.cfg.path_cost_scale_factor} Units")
        print(f"        > 满分奖励 (Prize Term): 100%覆盖 (1.0) = {self.cfg.prize_scale_factor} Units")
        
        # 自动计算并打印你刚才推导的逻辑
        break_even_dist = self.cfg.prize_scale_factor / self.cfg.path_cost_scale_factor
        print(f"        > 盈亏平衡点: 只有当距离 < {break_even_dist:.1f}米 时，才值得去一个满分点")
        
        max_p = np.max(prizes)
        print(f"        > 当前地图最高分节点: {max_p:.2f} (Scaled: {int(max_p * self.cfg.prize_scale_factor)})")
        # === [添加代码 END] ===


        for node_idx in range(1, num_nodes):
            routing.AddDisjunction(
                [manager.NodeToIndex(node_idx)], 
                int(scaled_prizes[node_idx]) 
            )

        # 添加路径预算约束
        # [修改]：因为成本矩阵已经是基于米计算并缩放的，
        # 所以预算只需： 米 * scale_factor
        budget_scaled = int(self.cfg.path_budget_meters * self.cfg.path_cost_scale_factor)
        
        print(f"    预算: {self.cfg.path_budget_meters}m -> Scaled Cost: {budget_scaled}")

        routing.AddDimension(
            transit_callback_index,
            0,              
            budget_scaled,  
            True,           
            'Distance'
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.FromSeconds(self.cfg.solver_time_limit_sec)

        solution = routing.SolveWithParameters(search_params)

        if solution:
            print("    [√] 成功找到解！")
            path_indices = []
            total_scaled_dist = 0
            
            index = routing.Start(0)
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                path_indices.append(node_idx)
                
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                total_scaled_dist += routing.GetArcCostForVehicle(prev_index, index, 0)
            
            path_indices.append(manager.IndexToNode(index))
            
            # 还原为实际米数
            real_dist_m = total_scaled_dist / self.cfg.path_cost_scale_factor
            collected_prize = sum(prizes[i] for i in path_indices)

            return path_indices, collected_prize, real_dist_m
        else:
            print("    [X] 未找到可行解。")
            return None, 0.0, 0.0

# ==========================================
# 3. 可视化模块 (坐标系适配)
# ==========================================

def visualize_results(game_map, all_nodes_m, path_indices, config: PlannerConfig):
    """
    可视化：
    - 背景是像素矩阵 (Pixel Space)
    - 点和路径是米 (Metric Space)
    我们需要将imshow的extent设置为物理尺寸，使它们对齐。
    """
    if path_indices is None:
        return

    optimal_nodes_m = all_nodes_m[path_indices]

    plt.figure(figsize=(10, 10))

    colors = ['#F0E68C', '#2F4F4F', '#FFC0CB'] 
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # [关键修改] 计算地图的物理尺寸 (米)
    map_h_px, map_w_px = game_map.shape
    map_w_m = map_w_px * config.pixel_resolution
    map_h_m = map_h_px * config.pixel_resolution

    # extent = [left, right, bottom, top]
    # 将像素图片映射到物理坐标系
    plt.imshow(game_map, cmap=cmap, origin='lower', 
               extent=[0, map_w_m, 0, map_h_m])

    # === [添加代码 START] 2. 画分界线 (Grid Lines) ===
    # 计算网格的物理步长 (例如 20px * 0.25 = 5米)
    step_m = config.grid_interval_pixels * config.pixel_resolution

    # 画垂直线
    for x in np.arange(0, map_w_m + step_m, step_m):
        plt.axvline(x, color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    # 画水平线
    for y in np.arange(0, map_h_m + step_m, step_m):
        plt.axhline(y, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    # === [添加代码 END] ===

    # === [修改代码] 1. 显示所有node，用小蓝点 ===
    # 将 c='gray' 改为 c='blue', alpha改高一点方便看清

    # 绘制候选节点 (已经是米了，直接画)
    plt.scatter(all_nodes_m[1:, 0], all_nodes_m[1:, 1], 
                c='blue', s=10, alpha=0.5, label='Candidate Nodes (Centroids)')

    # 绘制最优路径
    plt.plot(optimal_nodes_m[:, 0], optimal_nodes_m[:, 1], 
             'r-o', markersize=4, linewidth=2, label='Best Path')

    # 标记起点
    plt.plot(optimal_nodes_m[0, 0], optimal_nodes_m[0, 1], 
             'g*', markersize=15, label='Start (Depot)')

    plt.title(f"Planning Result (Standard Metric Units)\n"
              f"Budget: {config.path_budget_meters}m | Nodes are Centroids")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.3)

    # 反转Y轴
    plt.gca().invert_yaxis()

    plt.show()

# ==========================================
# 4. 主程序
# ==========================================

if __name__ == '__main__':
    # 配置：所有参数现在都是基于“米”进行调整，更加直观

    # Scaled Cost=Distance×100
    # Scaled Reward=100%×10000=10000
    # 求解器会在心里权衡：“我为了拿到这个奖励，最多愿意跑多远？” 答案是prize_scale_factor/path_cost_scale_factor


    config = PlannerConfig(
        pixel_resolution=0.25,      
        grid_interval_pixels=20,    
        fov_size_pixels=20,         
        path_budget_meters=1500.0,   # 1000米预算
        
        # 优化参数
        path_cost_scale_factor=1*100,      # 1米 = 100 cost
        prize_scale_factor=1*10000,   
        solver_time_limit_sec=5     
    )
    
    planner = GlobalPlanner(config)
    map_path = "/home/clp/workspace/SoViC/map/planning_maps/Area_2_map_1/Area_2_map_1_0.25m.npy"

    try:
        coral_map = planner.load_map(map_path)
        
        nodes_m, prizes = planner.generate_nodes_and_prizes(coral_map)
        cost_mat = planner.calculate_cost_matrix(nodes_m)
        path_indices, total_prize, dist_m = planner.solve_orienteering_problem(prizes, cost_mat)

        if path_indices is not None:
            print(f"\n[-] 最终结果:")
            print(f"    路径长度: {dist_m:.2f} m")
            print(f"    总奖励:   {total_prize:.4f}")
            visualize_results(coral_map, nodes_m, path_indices, config)
            
    except Exception as e:
        print(f"Error: {e}")