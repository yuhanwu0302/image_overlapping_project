import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('touch.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 1)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=4)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)


leaf1_edges = np.zeros_like(img)
leaf2_edges = np.zeros_like(img)

leaf1_edges[markers == 2] = 255
leaf2_edges[markers == 3] = 255

cv.imshow('touch1',img)
cv.imshow('Leaf1 Edges', leaf1_edges)
cv.imshow('Leaf2 Edges', leaf2_edges)
cv.waitKey(0)
cv.destroyAllWindows()

#  write each leaf
# cv.imwrite('leaf1_edges.jpg',leaf1_edges)
# cv.imwrite('leaf2_edges.jpg',leaf2_edges)
