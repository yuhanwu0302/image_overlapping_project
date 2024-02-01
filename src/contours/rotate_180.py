import cv2 as cv
import numpy as np
import os
from draw_contours import *
import pandas as pd
from tips_rotated import *

# # 定義需要旋轉的葉片編號
# need_rotation = set([1065, 1067, 1069, 1070, 1071, 1072, 1074, 1076, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1093, 1094, 1096,
#                      1100, 1102, 1103, 1106, 1107, 1111, 1112, 1113, 1114, 1117, 1118])
# ### 挑選
# Chinese_horse_chestnut = set(range(1060,1123))


targetdir =r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\5_true indigo\contourfiles" #r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\dataset_output\all_data\contourfiles"

output_dir=r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\grad\rotated_180" 

output_rotate_image_dir = r"C:\Users\Lab_205\Desktop\image_overlapping_project\src\contours\test_2\5_true indigo\rotated_imange\rotated_180" 

if not os.path.exists(output_rotate_image_dir):
     os.makedirs(output_rotate_image_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)




file_paths = get_all_file_paths(targetdir)
file_paths = [path for path in file_paths if path.endswith('.csv')]

for file_path in file_paths:
    # 解析文件名中的數字部分作為葉片編號
    filename = os.path.basename(file_path)
    leaf_index = int(filename.split('_')[0])

    #####挑選
    Chinese_horse_chestnut_id=int(filename.split('_')[0])
    
    
    # 判斷是否需要進行旋轉
    #if leaf_index in need_rotation:
        # 需要進行旋轉的葉片，進行逆時針旋轉 180 度
    points = read_contours(file_path)
    original_contours = np.array([(point.x, point.y) for point in points], dtype=np.float32)
    original_M = findCOM(file_path)
    original_points = [point.Point(x, y) for x, y in original_contours.tolist()]
    original_gradients = calculate_gradients(original_points)
    original_max_gradient = max([abs(g.value) for g in original_gradients])
    leaf_tip_candidate_index = [i for i, g in enumerate(original_gradients) if abs(g.value) == original_max_gradient]
    
    farthest_point = find_leaf_tip(original_contours, original_M)
    best_candidate_index = min(leaf_tip_candidate_index, 
    key=lambda i: np.linalg.norm(np.array([original_gradients[i].start_point.x, original_gradients[i].start_point.y]) - np.array(farthest_point)))


    leaf_tip = original_gradients[best_candidate_index].start_point
    leaf_tip = leaf_tip.x,leaf_tip.y

    angle = calculate_rotation_angle(leaf_tip,original_M)
    ##### 轉角度
    rotated = rotate_contour(original_contours,angle+180,original_M)
    adjusted_rotated = adjust_contour_position(rotated)
    clockwise = top_and_clockwise(adjusted_rotated)

    ###calculate rotated gradient###

    adjusted_rotated_points=[point.Point(x, y) for x, y in clockwise]
    adjusted_rotated_grad = calculate_gradients(adjusted_rotated_points,20,1)
    gradient_values = [gradient.value for gradient in adjusted_rotated_grad]
    
    base_name = os.path.basename(file_path)
    plotname, _ = os.path.splitext(base_name)


    rotated_counterclockwise_180 = draw(clockwise)
    
    output_csv_name = os.path.join(output_dir, f"{plotname}.csv")
    output_image_name = os.path.join(output_dir, f"{plotname}.jpg")
    
    output_gradvalue(gradient_values, output_image_name, output_csv_name)


    output_rotated_image_name = os.path.join(output_rotate_image_dir, f"{plotname}_rotated.jpg")
    output_rotated_csv_name = os.path.join(output_rotate_image_dir, f"{plotname}_rotated.csv")
    output_rotated_img(rotated_counterclockwise_180,output_rotated_image_name)
    output_rotated_img_csv(output_rotated_csv_name, clockwise)
    #else:
    #    pass


