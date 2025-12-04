import geopandas as gpd
import rasterio
from rasterio import features, transform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# --- 1. 定义文件路径和参数 ---
# !! 修改为您的Shapefile文件路径
map_index = 2
shp_path = f'./real_map/Dataset_3_Habitat_classifications/Area{map_index}/Protocol4_Reef_zones_and_geomorphometric_features.shp'

# !! 检查您的属性表，确定哪个字段包含分类信息
# !! 可能是 'CLASS_ID', 'DN', 'Gridcode', 'zone' 等
text_column = 'Class' # <--- !!! 在这里填入正确的列名 !!!

# 定义输出栅格的分辨率 (单位与shp的坐标系一致，例如米)
# 较小的值会产生更高分辨率的图像，但文件也更大
pixel_size = 0.1 

# 定义输出文件路径
output_npy_path = f'./output/segmentation_array_Area{map_index}.npy'
output_preview_path = f'./output/original_preview_Area{map_index}.png'


# --- 2. 读取Shapefile ---
print(f"Reading shapefile: {shp_path}")
gdf = gpd.read_file(shp_path)

# 检查数据 (可选)
# print("Shapefile head:")
# print(gdf.head())
# print(f"\nCoordinate Reference System (CRS): {gdf.crs}")
# print(f"Available columns: {gdf.columns.tolist()}")


gdf['CLASS_ID'] = gdf[text_column].astype('category').cat.codes + 1
classification_column = 'CLASS_ID'
category_mapping = dict(enumerate(gdf[text_column].astype('category').cat.categories, 1))
print("\nCategory to Integer ID Mapping:")
print(category_mapping)
# {1: 'Coral framework', 2: 'Macroalgae', 3: 'Rubble / Coral pavement', 4: 'Sand', 5: 'Seagrass', 6: 'unclassified'}


# --- 3. 定义颜色映射 ---
# 为每个类别定义颜色 (RGB值, 0-1范围)
color_mapping = {
    'Coral framework': '#FFC0CB',      # 珊瑚粉
    'Macroalgae': '#006400',           # 深绿色
    'Rubble / Coral pavement': '#2F4F4F',  # 暗灰色
    'Sand': '#F0E68C',                 # 卡其色
    'Seagrass': '#9ACD32',             # 黄绿色
    'unclassified': '#FFFFFF'          # 白色
}

# 创建颜色列表，按照CLASS_ID顺序
colors = ['#000000']  # 索引0为背景色(黑色)
for class_id, category_name in category_mapping.items():
    if category_name in color_mapping:
        colors.append(color_mapping[category_name])
    else:
        colors.append('#808080')  # 默认灰色

# 创建自定义colormap
custom_cmap = ListedColormap(colors)
# --- 4. 准备并执行栅格化 (后续代码与之前相同) ---
min_x, min_y, max_x, max_y = gdf.total_bounds
width = int((max_x - min_x) / pixel_size)
height = int((max_y - min_y) / pixel_size)
transform_matrix = transform.from_origin(min_x, max_y, pixel_size, pixel_size)

# 使用新的 classification_column ('CLASS_ID')
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[classification_column]))

print(f"\nCreating raster with dimensions (height, width): ({height}, {width})")

burned_array = features.rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform_matrix,
    fill=0,
    all_touched=True,
    dtype=np.uint8
)

print("Rasterization complete.")


# --- 5. 保存为numpy array ---
# 将栅格化结果保存为numpy array
segmentation_array = burned_array.copy()

# 创建输出目录（如果不存在）
import os
os.makedirs('./output', exist_ok=True)
np.save(output_npy_path, segmentation_array)
print(f"Segmentation array saved as numpy array: {output_npy_path}")
print(f"Array shape: {segmentation_array.shape}")
print(f"Unique class indices: {np.unique(segmentation_array)}")


# --- 6. 基于numpy array保存为可供预览的PNG图像 ---
plt.figure(figsize=(12, 10))

# 使用numpy array和自定义颜色映射
im = plt.imshow(segmentation_array, cmap=custom_cmap, vmin=0, vmax=len(colors)-1)

# # 创建颜色条并添加标签
# cbar = plt.colorbar(im, label='Habitat Classification')
# cbar.set_ticks(range(len(category_mapping) + 1))
# tick_labels = ['Background'] + [f"{class_id}: {name}" for class_id, name in category_mapping.items()]
# cbar.set_ticklabels(tick_labels)

plt.title('Coral Reef Habitat Classification Map')
plt.xlabel('X Coordinate (0.1m/pixel)')
plt.ylabel('Y Coordinate (0.1m/pixel)')

# 添加图例
legend_elements = []
for class_id, category_name in category_mapping.items():
    color = color_mapping.get(category_name, '#808080')
    legend_elements.append(Rectangle((0,0),1,1, facecolor=color, label=f'{category_name}'))

plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

plt.tight_layout()
plt.savefig(output_preview_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Preview image saved as PNG: {output_preview_path}")
print("\nColor mapping used:")
for class_id, category_name in category_mapping.items():
    color = color_mapping.get(category_name, '#808080')
    print(f"  {class_id}: {category_name} -> {color}")