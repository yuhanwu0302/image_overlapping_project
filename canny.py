import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("water_coins.jpg")
assert img is not None , "file could not be read"
edges = cv.Canny(img,180,200,apertureSize=5)

plt.subplot(121),plt.imshow(img,cmap="gray")
plt.title("Original image"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap="gray")
plt.title("Edge image"),plt.xticks([]),plt.yticks([])

plt.show()