import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("1079.jpg")
img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
img2 = cv.imread("1090.jpg")
img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

max_num_features = 500

orb = cv.ORB.create(max_num_features)
keypoint1 , descriptors1 = orb.detectAndCompute(img1_gray,None)
keypoint2 , descriptors2 = orb.detectAndCompute(img2_gray,None)




img1_display = cv.drawKeypoints(img1,keypoint1,outImage=
np.array([]),color=(255,0,0),flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img2_display = cv.drawKeypoints(img2,keypoint2,outImage=np.array([]),color=(255,0,0),flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=[20,10])
plt.subplot(121); plt.axis('off'); plt.imshow(img1_display); plt.title("Original Form");
plt.subplot(122); plt.axis('off'); plt.imshow(img2_display); plt.title("Scanned Form");                             

matcher = cv.DescriptorMatcher.create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = matcher.match(descriptors1,descriptors2,None)
matches = list(matches)
matches.sort(key=lambda x: x.distance, reverse=False)

numgoodmatches = int(len(matches)*0.001)
matches = matches[:8]

im_matches = cv.drawMatches(img1,keypoint1,img2,keypoint2,matches,None)
plt.figure(figsize=[40, 10])
plt.imshow(im_matches)
plt.axis("off")
plt.title("Original Form")


# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoint1[match.queryIdx].pt
    points2[i, :] = keypoint2[match.trainIdx].pt

# Find homography
h, mask = cv.findHomography(points2, points1, cv.RANSAC)

# Use homography to warp image
height, width, channels = img1.shape
img2_reg = cv.warpPerspective(img2, h, (width, height))

# Display results
plt.figure(figsize=[20, 10])
plt.subplot(121);plt.imshow(img1);    plt.axis("off");plt.title("Original Form")
plt.subplot(122);plt.imshow(img2_reg);plt.axis("off");plt.title("Scanned Form")