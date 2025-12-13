import numpy as np
from param import true_positive_rate, false_positive_rate
# Constants
L_MAX = 6.0    # P ~ 0.997
L_MIN = -6.0   # P ~ 0.002
L_PRIOR_ROCK = 0.0 # P = 0.5 (Unknown)
L_PRIOR_SAND = L_MIN # P ~ 0 (Impossible)

CONF_THRESHOLD = 0.8 

class BeliefMap:
    def __init__(self, map_size, cell_size, substrate_prior):
        """
        BeliefMap 初始化。
        :param substrate_prior: bool numpy array, True 表示是岩石(可能长珊瑚)，False 表示沙子。
        """
        self.map_size = map_size
        self.cell_size = cell_size
        
        # --- 1. Initialize Log-Odds based on Prior ---
        # 默认全部初始化为沙子 (L_MIN)
        self.log_odds = np.full(map_size, L_PRIOR_SAND, dtype=np.float32)
        
        # 将岩石区域初始化为未知 (0.0)
        self.log_odds[substrate_prior.astype(bool)] = L_PRIOR_ROCK
        
        # 概率图初始化
        self.probability_map = np.full(map_size, 0.0, dtype=np.float32)
        self.confirmation_map = np.zeros(map_size, dtype=bool)

        # 记录已经被相机【视觉确认】为珊瑚的位置
        # 这些位置虽然概率高，但Reward已被消耗，不应再作为探索目标
        self.visited_mask = np.zeros(map_size, dtype=bool)

        self._update_probabilities() # Sync initial prob map
        

    def update_belief(self, observations):
        """
        根据观测更新 Belief。
        关键点：只在 roi_mask (岩石) 区域进行更新。沙子区域永远保持 L_MIN。
        """
        
        # --- 1. Camera Update (High Confidence) ---
        cam_obs = observations.get('camera')
        if cam_obs and cam_obs['valid']:
            r1, r2, c1, c2 = cam_obs['bbox']
            
            # 获取局部 ROI (只关心岩石区域)
            # 因为我们假设沙子上肯定没珊瑚，就算相机看到空沙地，它本来就是L_MIN，不需要变
            # 只有当我们在岩石上看到空地时，才需要把 L_PRIOR_ROCK 降为 L_MIN
            local_log = self.log_odds[r1:r2, c1:c2]

            local_visited = self.visited_mask[r1:r2, c1:c2]

            
            # 1. Detected Coral (直接拉满)
            # 逻辑上 Coral 只能出现在 ROI 内，但为了鲁棒性，取交集
            mask_coral = cam_obs['detected_coral']
            local_log[mask_coral] = L_MAX
            local_visited[mask_coral] = True
            
            # 2. Detected Empty (直接拉低)
            # 只有在 ROI (岩石) 区域内看到空，才需要更新。沙子区域本来就是低。
            mask_empty = cam_obs['detected_empty'] 
            local_log[mask_empty] = L_MIN
            
            self.log_odds[r1:r2, c1:c2] = local_log
            self.visited_mask[r1:r2, c1:c2] = local_visited

        # --- 2. Sonar Update (Bayesian) ---
        sonar_obs = observations.get('sonar')
        if sonar_obs and sonar_obs['valid']:
            r1, r2, c1, c2 = sonar_obs['bbox']
            
            fov_mask = sonar_obs['mask']
            d_norm = sonar_obs['dists_norm']
            z_is_1 = sonar_obs['detections']
            
            # --- 关键：应用先验掩码 ---
            # 我们只更新位于 fov_mask 内的点
            valid_update_area = fov_mask
            
            # 如果视野里全是沙子，直接跳过计算，节省性能
            if np.any(valid_update_area):
                
                # Inverse Sensor Model 计算
                p_tp = true_positive_rate(d_norm)
                p_fp = false_positive_rate(d_norm)
                eps = 1e-6
                
                delta_l_detect = np.log((p_tp + eps) / (p_fp + eps))
                delta_l_miss = np.log((1 - p_tp + eps) / (1 - p_fp + eps))
                
                local_log = self.log_odds[r1:r2, c1:c2]
                update_val = np.zeros_like(local_log)
                
                # Update Rule:
                # 只在 valid_update_area (岩石) 上应用增量
                # 沙子区域 (False in valid_update_area) 保持 0 增量
                
                # Case 1: Detect (z=1) AND Valid Area
                mask_d = valid_update_area & z_is_1
                update_val[mask_d] = delta_l_detect[mask_d]
                
                # Case 2: Miss (z=0) AND Valid Area
                mask_m = valid_update_area & (~z_is_1)
                update_val[mask_m] = delta_l_miss[mask_m]
                
                # Apply
                local_log += update_val
                
                # Clamp (仅在岩石区域clamp即可，或者全图clamp也无妨)
                np.clip(local_log, L_MIN, L_MAX, out=local_log)
                
                self.log_odds[r1:r2, c1:c2] = local_log

        # --- 3. Sync Probability ---
        self._update_probabilities()

    def _update_probabilities(self):
        self.probability_map = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        # 2. 生成 Reward Field (Confirmation Map)
        # 条件 A: 概率 > 阈值 (包含声纳探测的高置信度区域 + 相机确认的区域)
        high_confidence = self.probability_map > CONF_THRESHOLD
        
        # 条件 B: 还没有被视觉确认过 (exclude visited corals)
        # 逻辑：是高置信度目标 AND (NOT 已经被相机看过并确认为珊瑚)
        self.confirmation_map = high_confidence & (~self.visited_mask)
        
    
    def get_snapshot(self):
        """
        创建当前 Belief 状态的快照。
        使用 .copy() 至关重要，这样在 Planning 计算期间（可能耗时），
        新的传感器更新不会修改规划器正在使用的矩阵。
        """
        snapshot = {
            # 1. 核心状态：Log-Odds (最原始的滤波数据)
            'log_odds': self.log_odds.copy(),
            
            # 2. 概率图：用于计算信息熵或直观展示
            'probability_map': self.probability_map.copy(),
            
            # 3. 奖励场：二值化地图，告诉规划器哪里值得去
            # (Valid Targets = High Prob AND Not Visited)
            'reward_map': self.confirmation_map.copy(),
            
            # 4. 辅助掩码：规划器可能需要知道哪里是岩石(可行域)或哪里去过了
            # 'roi_mask': self.roi_mask.copy(),      # 哪里可能长珊瑚
            # 'visited_coral_mask': self.visited_mask.copy() # 哪里已经拿过奖励了
        }
        return snapshot