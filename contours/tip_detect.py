import numpy as np
import glob
from scipy.interpolate import interp1d
from draw_contours import read_contours
import point
import cv2 as cv
import matplotlib.pyplot as plt

def findCOM(input_data):
    # 檢查輸入數據並讀取或轉換為輪廓數組
    if isinstance(input_data, str):  # 如果輸入是字符串，假設它是文件路徑
        points = read_contours(input_data)
        contour_points = [(point.x, point.y) for point in points]
        contour_array = np.array(contour_points, dtype=np.float32)
    elif isinstance(input_data, np.ndarray):  # 如果輸入是NumPy數組
        contour_array = input_data
    else:
        raise ValueError("Input must be a file path or a NumPy array.")

    # 計算輪廓的質心
    M = cv.moments(contour_array)
    if M['m00'] == 0:
        return (0, 0)  # 如果 m00 為 0，則返回 (0, 0) 避免除以零
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    return (cx, cy)  # 返回浮點數質心座標




def resample_contour(csv_file_path, target_num_points):
    points = read_contours(csv_file_path)
    contour = np.array([(point.x, point.y) for point in points], dtype=np.float32)

    arc_length = np.linspace(0, 1, len(contour))
    interp_func_x = interp1d(arc_length, contour[:, 0], kind='linear')
    interp_func_y = interp1d(arc_length, contour[:, 1], kind='linear')

    new_arc_length = np.linspace(0, 1, target_num_points)
    resampled_contour = np.column_stack((interp_func_x(new_arc_length), interp_func_y(new_arc_length)))

    return resampled_contour

def plot_contour(contour_points, title='Contour'):
    plt.figure(figsize=(6, 6))
    plt.plot(contour_points[:, 0], contour_points[:, 1], linewidth=2)
    plt.title(title)
    plt.axis('equal')
    plt.show()


def plot_points(contour_points, title="Points"):
    plt.figure(figsize=(6, 6))
    plt.scatter(contour_points[:, 0], contour_points[:, 1], s=10)
    plt.title(title)
    plt.axis('equal')
    plt.show()

def find_leaf_tip(input_data, centroid):
    if isinstance(input_data, str):  # 如果输入是字符串，假设它是文件路径
        points = read_contours(input_data)
        contour_points = [(point.x, point.y) for point in points]
    elif isinstance(input_data, np.ndarray):  # 如果输入是NumPy数组
        contour_points = input_data
    else:
        raise ValueError("Input must be a file path or a NumPy array.")

    distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for point in contour_points]
    tip_index = np.argmax(distances)
    return contour_points[tip_index]


def plot_comparison(original_contour, resampled_contour, original_centroid, resampled_centroid, original_tip, resampled_tip):
    plt.figure(figsize=(10, 10))

    # 繪製原始輪廓
    plt.plot(original_contour[:, 0], original_contour[:, 1], 'b-', label='Original Contour')

    # 繪製重採樣後的輪廓
    plt.plot(resampled_contour[:, 0], resampled_contour[:, 1], 'r--', label='Resampled Contour')

    # 標記原始質心
    plt.plot(original_centroid[0], original_centroid[1], 'bo', label='Original Centroid')

    # 標記重採樣後的質心
    plt.plot(resampled_centroid[0], resampled_centroid[1], 'ro', label='Resampled Centroid')

    # 標記原始葉尖
    plt.plot(original_tip[0], original_tip[1], 'go', label='Original Leaf Tip')

    # 標記重採樣後的葉尖
    plt.plot(resampled_tip[0], resampled_tip[1], 'mo', label='Resampled Leaf Tip')

    plt.title('Comparison of Original and Resampled Contours')
    plt.legend()
    plt.axis('equal')
    plt.show()

#  calculate all files 
#  csv_files = glob.glob(r'C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01/*.csv')

#############################   test data    ######################
csv_files =r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\1060_clear.csv"
target_num_points = 100
resampled_contours = resample_contour(csv_files, target_num_points)

##  original contour
points = read_contours(csv_files)
original_contours = np.array([(point.x, point.y) for point in points], dtype=np.float32)
original_M =findCOM(csv_files)

##  插值後的M
contour_int = np.round(resampled_contours).astype(np.int32).reshape((1, -1, 2))
M = cv.moments(contour_int)
cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
resample_M = (cx, cy)

print(original_M)
print(resample_M)

plot_contour(original_contours)
plot_contour(resampled_contours)

plot_points(original_contours)
plot_points(resampled_contours)

original_tip  =  find_leaf_tip(original_contours,original_M)
resampled_tip =  find_leaf_tip(resampled_contours,resample_M)

plot_comparison(original_contours,resampled_contours,original_M,resample_M,original_tip,resampled_tip)






# Procrustes test

from scipy.spatial import procrustes



file1 = r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\1060_clear.csv"
file2 = r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\2_Chinese horse chestnut\contourfiles01\1061_clear.csv"
points1 = read_contours(file1)
contours1 = np.array([(point.x, point.y) for point in points1], dtype=np.float32)

points2 = read_contours(file2)
contours2 = np.array([(point.x, point.y) for point in points2], dtype=np.float32)

target_num_points = 100
resampled_contours1  = resample_contour(file1,target_num_points)
resampled_contours2  = resample_contour(file2,target_num_points)



mtx_a, mtx_b, disparity = procrustes(resampled_contours1, resampled_contours2)


procrustes_COM_1= findCOM(mtx_a)
procrustes_COM_2 = findCOM(mtx_b)
procrustes_tip_1 = find_leaf_tip(mtx_a, procrustes_COM_1)
procrustes_tip_2 = find_leaf_tip(mtx_b, procrustes_COM_2)

def plot_aligned_contours(contour1, contour2, centroid1, centroid2, tip1, tip2, title='Aligned Contours Comparison'):
    plt.figure(figsize=(10, 5))


    plt.subplot(1, 2, 1)
    plt.plot(contour1[:, 0], contour1[:, 1], 'r-', label='Contour 1')
    plt.scatter(centroid1[0], centroid1[1], c='k', marker='o', label='Centroid 1')
    plt.scatter(tip1[0], tip1[1], c='g', marker='x', label='Tip 1')
    plt.title('Contour 1')
    plt.axis('equal')

 
    plt.subplot(1, 2, 2)
    plt.plot(contour2[:, 0], contour2[:, 1], 'b-', label='Contour 2')
    plt.scatter(centroid2[0], centroid2[1], c='k', marker='o', label='Centroid 2')
    plt.scatter(tip2[0], tip2[1], c='g', marker='x', label='Tip 2')
    plt.title('Contour 2')
    plt.axis('equal')

    plt.suptitle(title)
    plt.legend()
    plt.show()


plot_aligned_contours(mtx_a, mtx_b, procrustes_COM_1, procrustes_COM_2, procrustes_tip_1, procrustes_tip_2)
