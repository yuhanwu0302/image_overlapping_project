import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import draw_contours as dc


def set_starting_point_to_leftmost(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    
    min_index = np.argmin(x)
    
    #最左邊的點成為起點
    reordered_contour = np.roll(contour, -min_index, axis=0)
    
    return reordered_contour

testdata_1 = np.zeros((512,512,3),np.uint8)
pts = np.array([[128,328], [64,455], [128,510], [192, 455]])
cv.polylines(testdata_1, [pts], True, (255,255,255))
gray = cv.cvtColor(testdata_1,cv.COLOR_BGR2GRAY)
_ ,thresh = cv.threshold(gray,0,255,cv.THRESH_TOZERO+cv.THRESH_OTSU)
contours,hierarchy=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
reordered_contours = [set_starting_point_to_leftmost(contour) for contour in contours]




outputname = "rota1.csv"
with open(outputname, "w") as f:
    for layer_1 in reordered_contours:
        for layer_2 in layer_1:
            f.writelines([f"{pixel}," for pixel in layer_2])
            f.writelines("\n")



testdata_2 = np.zeros((512,512,3),np.uint8)
pts = np.array([[256,128], [128,256], [256,500], [384, 256]])
cv.polylines(testdata_2, [pts], True, (255,255,255))
gray = cv.cvtColor(testdata_2,cv.COLOR_BGR2GRAY)
_ ,thresh = cv.threshold(gray,0,255,cv.THRESH_TOZERO+cv.THRESH_OTSU)
contours,hierarchy=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
reordered_contours = [set_starting_point_to_leftmost(contour) for contour in contours]

outputname2 = "rota2.csv"
with open(outputname2, "w") as f:
    for layer_1 in reordered_contours:
        for layer_2 in layer_1:
            f.writelines([f"{pixel}," for pixel in layer_2])
            f.writelines("\n")


""" 旋轉輪廓
center = tuple(np.mean(pts, axis=0))

# 创建旋转矩阵（逆时针旋转 35 度）
rotation_matrix = cv.getRotationMatrix2D(center, -70, 1.0)

# 由于 cv2.warpAffine 作用于图像，对于单独的点集，我们使用另一种方法
# 将点集转换为齐次坐标以便应用仿射变换
ones = np.ones(shape=(len(pts), 1))
pts_homogeneous = np.hstack([pts, ones])

# 应用旋转矩阵
rotated_pts = rotation_matrix.dot(pts_homogeneous.T).T

# 打印旋转后的点
print(rotated_pts)
"""


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(testdata_1)
plt.title('testdata_1')

plt.subplot(1, 2, 2)
plt.imshow(testdata_2)
plt.title('testdata_2')

plt.show()


