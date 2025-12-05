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

GAME_FPS = 4

# manual control:
KEY_VELOCITY = 1.0  # m/s
KEY_ANGULAR_VELOCITY = 1.0  # rad/s

MAP_DIR = "./map/planning_maps/Area_2_map_1/Area_2_map_1_0.25m.npy"
