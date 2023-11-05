import numpy as np
import glob
from scipy.interpolate import interp1d
from draw_contours import read_contours
import cv2 as cv
import matplotlib.pyplot as plt

def findCOM(contourfile):
    points = read_contours(contourfile)
    contour_points = [(point.x, point.y) for point in points]
    contour_array = np.array(contour_points, dtype=np.float32)
    M = cv.moments(contour_array)
    cx = int(M['m10']/M['m00']) if M['m00'] != 0 else 0
    cy = int(M['m01']/M['m00']) if M['m00'] != 0 else 0
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

def find_leaf_tip(contour_points, centroid):
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

##############################   test data    ######################
csv_files =r"C:\Users\Lab_205\Desktop\image_overlapping_project\dataset_output\find_pattern\7_Nanmu\contourfiles01\1339_clear.csv"
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