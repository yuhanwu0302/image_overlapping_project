import cv2
import numpy as np
import csv
import re

# 從CSV文件中讀取座標點
points = []
with open(r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\csvfiles\3259_clear.csv', 'r') as file:
    reader = csv.reader(file, delimiter=' ')
    for row in reader:
        match = re.match(r'\[(\d+)\s+(\d+)\]', ' '.join(row))
        if match:
            x, y = map(int, match.groups())
            points.append((x, y))

# 創建白色畫布
height, width = 500, 500  # 根據需要調整畫布的大小
canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

# 繪製每一對相鄰的點
for i in range(1, len(points)):
    cv2.line(canvas, points[i-1], points[i], (0, 0, 0), 1)  # 使用黑色線條
    cv2.imshow("Drawing", canvas)
    cv2.waitKey(1)  # 每繪製一條線後等待100毫秒

cv2.waitKey(0)
cv2.destroyAllWindows()










### test 
import cv2 as cv




def set_starting_point_to_leftmost(contour):
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    
    min_index = np.argmin(x)
    
    #最左邊的點成為起點
    reordered_contour = np.roll(contour, -min_index, axis=0)
    
    return reordered_contour

# 讀取圖像
img = cv.imread(r'C:\Users\Lab_205\Desktop\image_processing_opencv\plant leaf\Flavia dataset\clear\1425_clear.jpg')

# 轉換為灰度圖像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 使用二值化來獲得二值圖像
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# 找到輪廓
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
reordered_contours = [set_starting_point_to_leftmost(contour) for contour in contours]
with open("test_contours.csv", "w") as f:
    for layer_1 in reordered_contours:
        for layer_2 in layer_1:
            f.writelines([f"{pixel}," for pixel in layer_2])
            f.writelines("\n")

# 繪製輪廓
# -1 表示繪製所有輪廓
# (0, 255, 0) 是繪製輪廓的顏色 (在這裡是綠色)
# 2 是線的寬度
cv.drawContours(img, contours, -1, (0, 255, 0), 2)

# 顯示圖像
cv.imshow('Contours', img)
cv.waitKey(0)
cv.destroyAllWindows()
