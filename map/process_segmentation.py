import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import os

# 设置随机种子以确保结果可重复
random.seed(42)
np.random.seed(42)

# 注意：输入的numpy array分辨率是0.1m/pixel
# 此代码会将其下采样到0.25m/pixel，这样一个像素正好代表一个珊瑚的大小

# --- 1. 参数设置 ---
area_index = 2
map_index = 1
# 定义区域范围 (基于原始0.1m/pixel分辨率)
range_x = (1000, 2000)  # x范围 (0.1m/pixel下100-200m的区域)
range_y = (1500, 2500)  # y范围 (0.1m/pixel下150-250m的区域)

input_npy_path = f'./output/segmentation_array_Area{area_index}.npy'
# 保存到规划地图专用目录
area_folder = f'./planning_maps/Area_{area_index}_map_{map_index}/'
os.makedirs(area_folder, exist_ok=True)
planning_map_npy_path = os.path.join(area_folder, f'Area_{area_index}_map_{map_index}_0.25m.npy')
planning_map_preview_path = os.path.join(area_folder, f'Area_{area_index}_map_{map_index}_0.25m.png')
info_txt_path = os.path.join(area_folder, 'info.txt')


# 下采样因子：从0.1m/pixel到0.25m/pixel需要2.5倍下采样
# 我们使用精确的2.5倍下采样来达到0.25m/pixel
downsample_factor = 2.5  # 精确的0.25m/pixel

# class1转换为class3的概率 (现在一个pixel代表一个珊瑚，所以概率可以更高)
coral_spot_probability = 0.2  # 20%的概率将class1的像素转换为class3（单个珊瑚）

# --- 2. 读取原始numpy array ---
# 创建规划地图目录（如果不存在）
print(f"Loading segmentation array from: {input_npy_path}")
original_array = np.load(input_npy_path)
print(f"Original array shape: {original_array.shape}")
print(f"Original unique classes: {np.unique(original_array)}")

# --- 3. 提取指定区域 ---
x_start, x_end = range_x
y_start, y_end = range_y

# 确保范围不超出数组边界
x_start = max(0, x_start)
x_end = min(original_array.shape[1], x_end)
y_start = max(0, y_start)
y_end = min(original_array.shape[0], y_end)

print(f"Extracting region: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
cropped_array = original_array[y_start:y_end, x_start:x_end].copy()
print(f"Cropped array shape: {cropped_array.shape}")

# --- 4. 下采样到精确的0.25m/pixel ---
def downsample_array(arr, factor):
    """下采样数组，支持非整数因子，使用最常见值作为新像素值"""
    new_height = int(arr.shape[0] / factor)
    new_width = int(arr.shape[1] / factor)
    downsampled = np.zeros((new_height, new_width), dtype=arr.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            # 计算在原数组中的位置范围
            y_start = int(i * factor)
            y_end = int((i + 1) * factor)
            x_start = int(j * factor)
            x_end = int((j + 1) * factor)
            
            # 确保不超出边界
            y_end = min(y_end, arr.shape[0])
            x_end = min(x_end, arr.shape[1])
            
            # 提取对应的块
            block = arr[y_start:y_end, x_start:x_end]
            if block.size > 0:
                # 使用最常见的值
                values, counts = np.unique(block, return_counts=True)
                downsampled[i, j] = values[np.argmax(counts)]
    
    return downsampled

print(f"Downsampling by factor {downsample_factor} (0.1m -> 0.25m per pixel)")
cropped_array = downsample_array(cropped_array, downsample_factor)
print(f"Downsampled array shape: {cropped_array.shape}")

# --- 5. 重新分类 ---
processed_array = np.zeros_like(cropped_array)

# 创建类别映射
# 原class 1,2,3 -> 新class 1 (岩石)
# 原class 4,5,6 -> 新class 0 (背景)
# 部分原class 1 -> 新class 2 (珊瑚点)

for i in range(cropped_array.shape[0]):
    for j in range(cropped_array.shape[1]):
        original_class = cropped_array[i, j]
        
        if original_class in [1, 2, 3]:  # 原来的class 1,2,3
            # 如果是class 1，有一定概率转换为class 2（珊瑚点）
            if original_class == 1 and random.random() < coral_spot_probability:
                processed_array[i, j] = 2  # 珊瑚点
            else:
                processed_array[i, j] = 1  # 岩石
        elif original_class in [4, 5, 6]:  # 原来的class 4,5,6
            processed_array[i, j] = 0  # 背景
        else:
            processed_array[i, j] = 0  # 背景保持不变

print(f"Processed array unique classes: {np.unique(processed_array)}")

# --- 6. 保存处理后的numpy array ---
np.save(planning_map_npy_path, processed_array)
print(f"Processed array saved as: {planning_map_npy_path}")

# --- 7. 创建可视化 ---
# 定义新的颜色映射
colors = [
    '#F0E68C',  # 0: 背景sand沙 (卡其色)
    '#2F4F4F',  # 1: 岩石 (褐色)
    '#FFC0CB'   # 2: 珊瑚点 (粉色)
]

custom_cmap = ListedColormap(colors)

plt.figure(figsize=(12, 10))
im = plt.imshow(processed_array, cmap=custom_cmap, vmin=0, vmax=2)

# 设置实际米数刻度
# 计算实际覆盖的距离（以米为单位）
actual_width_m = (range_x[1] - range_x[0]) * 0.1  # 原始分辨率0.1m/pixel
actual_height_m = (range_y[1] - range_y[0]) * 0.1  # 原始分辨率0.1m/pixel

# 设置坐标轴刻度为实际米数
x_ticks = np.linspace(0, processed_array.shape[1]-1, 6)  # 6个刻度点
y_ticks = np.linspace(0, processed_array.shape[0]-1, 6)  # 6个刻度点
x_labels = [f"{int(x)}m" for x in np.linspace(0, actual_width_m, 6)]  # 对应的米数标签
y_labels = [f"{int(y)}m" for y in np.linspace(0, actual_height_m, 6)]  # 对应的米数标签

plt.xticks(x_ticks, x_labels)
plt.yticks(y_ticks, y_labels)

# # 添加颜色条
# cbar = plt.colorbar(im, label='Processed Classification')
# cbar.set_ticks([0, 1, 2, 3])
# cbar.set_ticklabels(['Background', 'Rock', 'Sand', 'Coral Spots'])

plt.title(f'Processed Coral Reef Map - Area {map_index}\nResolution: 0.25m/pixel, Coverage: {actual_width_m:.0f}m × {actual_height_m:.0f}m')
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')

# 添加图例
from matplotlib.patches import Rectangle
legend_elements = [
    Rectangle((0,0),1,1, facecolor='#2F4F4F', label='Rock (former classes 1,2,3)'),
    Rectangle((0,0),1,1, facecolor='#F0E68C', label='Sand (former classes 4,5,6)'),
    Rectangle((0,0),1,1, facecolor='#FFC0CB', label='Individual Corals (0.25m each)')
]

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig(planning_map_preview_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Processed preview image saved as: {planning_map_preview_path}")

# --- 保存统计信息到文本文件 ---
# 确保统计变量在保存时已定义
with open(info_txt_path, 'w') as info_file:
    info_file.write("=== Processing Statistics ===\n")
    info_file.write(f"Total pixels: {processed_array.size} (each pixel = 0.25m x 0.25m = 0.0625m²)\n")
    info_file.write(f"Background: {np.sum(processed_array == 0)} ({np.sum(processed_array == 0)/processed_array.size*100:.1f}%)\n")
    info_file.write(f"Rock: {np.sum(processed_array == 1)} ({np.sum(processed_array == 1)/processed_array.size*100:.1f}%)\n")
    info_file.write(f"Sand: {np.sum(processed_array == 2)} ({np.sum(processed_array == 2)/processed_array.size*100:.1f}%)\n")
    info_file.write(f"Individual corals: {np.sum(processed_array == 3)} ({np.sum(processed_array == 3)/processed_array.size*100:.1f}%) - {np.sum(processed_array == 3)} corals total\n")
    info_file.write(f"Original resolution: 0.1m/pixel -> Downsampled to: 0.25m/pixel (factor {downsample_factor})\n")
    info_file.write(f"Total area covered: {processed_array.size * 0.0625:.1f} m²\n")
    info_file.write(f"Range X(pixel): {range_x}\n")
    info_file.write(f"Range Y(pixel): {range_y}\n")
    info_file.write(f"Range X(m): {range_x[0]*0.1} - {range_x[1]*0.1}\n")
    info_file.write(f"Range Y(m): {range_y[0]*0.1} - {range_y[1]*0.1}\n")


print(f"Processing statistics saved as: {info_txt_path}")
