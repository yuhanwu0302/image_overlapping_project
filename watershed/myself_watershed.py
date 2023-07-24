import cv2 as cv 
import numpy as np
img = cv.imread("water_coins.jpg")
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("coin",gray)
cv.waitKey()
cv.destroyAllWindows()

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, bg_markers = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
cv.imshow('image', bg_markers)
cv.waitKey(0)
cv.destroyAllWindows()
sure_fg = cv.erode(opening, kernel, iterations=3)
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, fg_markers = cv.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

cv.imshow('image', fg_markers)
cv.waitKey(0)
cv.destroyAllWindows()



markers = cv.subtract(bg_markers, fg_markers)
markers = markers.astype(np.int32)
markers = cv.watershed(img, markers)
markers = markers.astype(np.uint8)

img[markers == -1] = [0, 0, 255]
overlay = cv.addWeighted(img, 0.5, cv.cvtColor(thresh, cv.COLOR_GRAY2BGR), 0.5, 0)

cv.imshow('image', overlay)
cv.waitKey(0)
cv.destroyAllWindows()


