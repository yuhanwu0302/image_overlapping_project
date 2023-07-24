from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2 as cv 

image = cv.imread(r"C:\Users\baba\Desktop\image_processing_opencv\watershed\water_coins.jpg")
shifted = cv.pyrMeanShiftFiltering(image, 21, 51)
cv.imshow("Input", image)
cv.waitKey(0)
cv.destroyAllWindows()

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255,
	cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
cv.imshow("Thresh", thresh)
cv.waitKey(0)
cv.destroyAllWindows()



D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, min_distance=20,labels=thresh)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv.minEnclosingCircle(c)
	cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
		cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image
cv.imshow("Output", image)
cv.waitKey(0)