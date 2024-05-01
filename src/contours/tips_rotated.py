import numpy as np
import glob
import contours.point
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from contours.mark import plot_mark_contours,plot_gradients
from contours.draw_contours import *
import pandas as pd
import contours.point
del globals()['main']


def findCOM(input_data):
    if isinstance(input_data, str): 
        points = read_contours(input_data)
        contour_points = [(point.x, point.y) for point in points]
        contour_array = np.array(contour_points, dtype=np.float32)
    elif isinstance(input_data, np.ndarray):
        contour_array = input_data
    else:
        raise ValueError("Input must be a file path or a NumPy array.")

    M = cv.moments(contour_array)
    if M['m00'] == 0:
        return (0, 0)  
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return (cx, cy)  

def resample_contour(csv_file_path, target_num_points):
    points = read_contours(csv_file_path)
    contour = np.array([(point.x, point.y) for point in points], dtype=np.float32)

    arc_length = np.linspace(0, 1, len(contour))
    interp_func_x = interp1d(arc_length, contour[:, 0], kind='linear')
    interp_func_y = interp1d(arc_length, contour[:, 1], kind='linear')

    new_arc_length = np.linspace(0, 1, target_num_points)
    resampled_contour = np.column_stack((interp_func_x(new_arc_length), interp_func_y(new_arc_length)))

    return resampled_contour

def find_leaf_tip(input_data, centroid):
    if isinstance(input_data, str):
        points = read_contours(input_data)
        contour_points = [(point.x, point.y) for point in points]
    elif isinstance(input_data, np.ndarray):  
        contour_points = input_data
    else:
        raise ValueError("Input must be a file path or a NumPy array.")

    distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for point in contour_points]
    tip_index = np.argmax(distances)
    return contour_points[tip_index]

def calculate_rotation_angle(tip, centroid):
    # 計算角度
    vector_to_tip = np.array(tip) - np.array(centroid)
    angle_with_vertical = np.arctan2(vector_to_tip[0], -vector_to_tip[1])
    return np.degrees(angle_with_vertical)


def rotate_contour(contour, angle, centroid):
    # 添加一列1以構成齊次座標
    ones = np.ones((contour.shape[0], 1))
    contour_homogeneous = np.hstack((contour, ones))

    # 計算旋轉矩陣
    rotation_matrix = cv.getRotationMatrix2D(centroid, angle, 1)

    # 應用彷射變換
    rotated_contour = cv.transform(np.array([contour_homogeneous]), rotation_matrix)
    rotated_contour_int = rotated_contour[0][:, :2].astype(int)
    # 返回變換後的輪廓，刪除齊次座標的最後一列
    return rotated_contour_int


def draw(contour, size=(800, 600)):
    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    contour_int = np.round(contour).astype(np.int32)
    canvas=cv.polylines(canvas, [contour_int], isClosed=True, color=(255, 255, 255), thickness=2)
    return canvas


def adjust_contour_position(contour, size=(800, 600)):
    x_min, y_min = np.min(contour, axis=0)
    x_max, y_max = np.max(contour, axis=0)

    # 計算平移距離
    dx = min(0, size[1] - x_max) if x_max > size[1] else -min(0, x_min)
    dy = min(0, size[0] - y_max) if y_max > size[0] else -min(0, y_min)

    # 平移
    translated_contour = contour + np.array([dx, dy])
    return translated_contour

def top_and_clockwise(contour):
    x = contour[:, 0]
    y = contour[:, 1]
    min_y_index = np.argmin(y)

    reordered_contour = np.roll(contour, -min_y_index, axis=0)

    area = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if area < 0:
        reordered_contour = np.flipud(reordered_contour)

    return reordered_contour

def output_rotated_img(img,output_rotated_image_name:str,):
    cv.imwrite(f"{output_rotated_image_name}",img)

def output_rotated_img_csv(output_csv_name:str,retaed_contour):
    df = pd.DataFrame(retaed_contour)
    df.to_csv(output_csv_name, index=False,encoding="utf-8",header=False)
    

#############################   run data    ######################
def main():       
    targetdir = r'C:\Users\baba\Desktop\image_overlapping_project\dataset_output\test_rotated_vs_unrotated\clear\contourfiles'
    output_dir = r'C:\Users\baba\Desktop\image_overlapping_project\dataset_output\test_rotated_vs_unrotated\clear\contourfiles\new1'
    
    output_rotate_image_dir= r'C:\Users\baba\Desktop\image_overlapping_project\dataset_output\test_rotated_vs_unrotated\clear\contourfiles\new1\img'

    if not os.path.exists(output_rotate_image_dir):
        os.makedirs(output_rotate_image_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = get_all_file_paths(targetdir)
    for csvfile in all_files:
        points = read_contours(csvfile)
        original_contours = np.array([(point.x, point.y) for point in points], dtype=np.float32)
        original_M = findCOM(csvfile)
        original_points = [Point(x, y) for x, y in original_contours.tolist()]
        original_gradients = calculate_gradients(original_points)
        original_max_gradient = max([abs(g.value) for g in original_gradients])
        leaf_tip_candidate_index = [i for i, g in enumerate(original_gradients) if abs(g.value) == original_max_gradient]

        farthest_point = find_leaf_tip(original_contours, original_M)
        best_candidate_index = min(leaf_tip_candidate_index, key=lambda i: np.linalg.norm(np.array([original_gradients[i].start_point.x, original_gradients[i].start_point.y]) - np.array(farthest_point)))

        leaf_tip = original_gradients[best_candidate_index].start_point
        leaf_tip = leaf_tip.x,leaf_tip.y

        angle = calculate_rotation_angle(leaf_tip,original_M)
        rotated = rotate_contour(original_contours,angle+90,original_M)
        adjusted_rotated = adjust_contour_position(rotated)
        clockwise = top_and_clockwise(adjusted_rotated)

        ###calculate rotated gradient###

        adjusted_rotated_points=[Point(x, y) for x, y in clockwise]
        adjusted_rotated_grad = calculate_gradients(adjusted_rotated_points,20,1)
        gradient_values = [gradient.value for gradient in adjusted_rotated_grad]
        
        base_name = os.path.basename(csvfile)
        plotname, _ = os.path.splitext(base_name)
        
        output_csv_name = os.path.join(output_dir, f"{plotname}.csv")
        output_image_name = os.path.join(output_dir, f"{plotname}.jpg")
        output_rotated_image_name = os.path.join(output_rotate_image_dir, f"{plotname}_rotated.jpg")
        output_rotated_csv_name = os.path.join(output_rotate_image_dir, f"{plotname}_rotated.csv")
        output_gradvalue(gradient_values, output_image_name, output_csv_name)
        rotaed_img = draw(clockwise)
        output_rotated_img(rotaed_img,output_rotated_image_name)
        output_rotated_img_csv(output_rotated_csv_name,clockwise)

        second_derivatives = calculate_second_derivatives(adjusted_rotated_grad,1)
        second_derivative_values = [derivative for derivative in second_derivatives]
        output_second_derivative(second_derivative_values, output_image_name, output_csv_name)


'''check which part is which part
interval = 10
move = 1
points = adjusted_rotated_points
gradients = adjusted_rotated_grad
plot_mark_contours(points, gradients, interval, move,70,80)
plot_gradients(gradients, interval, move,70,80)
'''



'''
# 查看原本輪廓與翻轉後的輪廓
rotated_contour_image = draw(adjusted_rotated)
original_contour_image = draw(original_contours)
rotated_contour_image = draw(clockwise)
cv.imshow('Original Contour', original_contour_image)
cv.imshow('Rotated Contour', rotated_contour_image)
cv.waitKey(0)
cv.destroyAllWindows()
'''


'''
# 查看輪廓 之後可以import dynamic.py
def run(contours):
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 800)
    ax.invert_yaxis()
    # 定义按键响应函数
    def on_press(event):
        if event.key == 'escape':  # 如果按下的是 Esc 键
            plt.close(fig)  # 关闭图表窗口

    # 将按键事件与处理函数绑定
    fig.canvas.mpl_connect('key_press_event', on_press)

    # 逐点绘制轮廓
    for i in range(len(contours)):
        x, y = clockwise[i, 0], clockwise[i, 1]
        ax.scatter(x, y)
        plt.pause(0.000001)
        if not plt.fignum_exists(fig.number):  # 如果图表已关闭，则结束循环
            break

    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()

run(clockwise)
'''


