import sys
sys.path.insert(0,'../libs/OpencvToolsKit')
from tools import *
from keypoint_detectors import *
import os
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

image = cv2.imread(filename= '../original_money/100k_back.jpg',flags= cv2.IMREAD_GRAYSCALE)
#cv2.namedWindow('100.000 VND back', cv2.WINDOW_NORMAL)
#cv2.imshow('100.000 VND back',frame_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# BRISK descriptor

BRISK = cv2.BRISK_create()
keypoints_BRISK = BRISK.detect(image, None)

print("Number of keypoints detected: ", len(keypoints_BRISK))

# Save Keypoints to a filename
index = []
for point in keypoints_BRISK:
    temp = (point.pt,point.size,point.angle,point.response,point.octave,point.class_id)
    index.append(temp)

# Keypoints file name
kfn = "../original_money/BRISK-100k-back-keypoints.txt"
# Delete a file if it exists
if os.path.exists(kfn):
    os.remove(kfn)

# Open a file
file = open(kfn, "wb")

# Write 
file.write(pickle.dumps(index))

# Close a file
file.close()


# Compute the descriptors with BRISK
keypoints_BRISK, descriptors_BRISK = BRISK.compute(image, keypoints_BRISK[:])

# Print the descriptor size in bytes
print("Size of Descriptor:", BRISK.descriptorSize(), "\n")

# Print the descriptor type
print("Type of Descriptor:", BRISK.descriptorType(), "\n")

# Print the default norm type
print("Default Norm Type:", BRISK.defaultNorm(), "\n")

# Print shape of descriptor
print("Shape of Descriptor:", descriptors_BRISK.shape, "\n")

## Draw only 50 keypoints on input image
#image = cv2.drawKeypoints(image = image,
#                         keypoints = keypoints_BRISK[:],
#                         outImage = None,
#                         flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

## Plot input image

## Turn interactive plotting off
#plt.ioff()

## Create a new figure
#plt.figure()
#plt.axis('off')
#plt.imshow(image)
#plt.show()

#plt.imsave(fname = 'feature-detection-BRISK.png',
#           arr = image,
#           dpi = 300)

## Close it
#plt.close()


# create BFMatcher object
# BFMatcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
#                          crossCheck = True)
# cap = cv2.VideoCapture('../video_test/100k.MOV')
# fps = cap.get(cv2.CAP_PROP_FPS)

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_keypoints, frame_descriptors = BRISK.detectAndCompute(frame, None)

#     # Matching descriptor vectors using Brute Force Matcher
#     matches = BFMatcher.match(queryDescriptors = descriptors_BRISK,
#                               trainDescriptors = frame_descriptors)
#     # Sort them in the order of their distance
#     matches = sorted(matches, key = lambda x: x.distance)
    
#     # Draw first 15 matches
#     output = cv2.drawMatches(img1 = image,
#                             keypoints1 = keypoints_BRISK,
#                             img2 = frame,
#                             keypoints2 = frame_keypoints,
#                             matches1to2 = matches[:15],
#                             outImg = None,
#                             flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     cv2.namedWindow("output", cv2.WINDOW_NORMAL)     
#     cv2.imshow('output',output)

 
#     if cv2.waitKey(1000) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# FLANN parameters
FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 5)

search_params = dict(checks = 50)

# Converto to float32
descriptors1 = np.float32(descriptors_BRISK)

cap = cv2.VideoCapture('../video_test/100k.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)

FLANN = cv2.FlannBasedMatcher(indexParams = index_params,searchParams = search_params)

while(cap.isOpened()):
	ret, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_keypoints, frame_descriptors = BRISK.detectAndCompute(frame, None)

	descriptors2 = np.float32(frame_descriptors)
	# Create FLANN object


	# Matching descriptor vectors using FLANN Matcher
	matches = FLANN.knnMatch(queryDescriptors = descriptors1,trainDescriptors = descriptors2,k = 2)

	# Lowe's ratio test
	ratio_thresh = 0.7

	# "Good" matches
	good_matches = []

	for m, n in matches:
		if m.distance < ratio_thresh * n.distance:
			good_matches.append(m)

	# Draw only "good" matches
	output = cv2.drawMatches(img1 = image,keypoints1 = keypoints_BRISK,img2 = frame,keypoints2 = frame_keypoints,matches1to2 = good_matches,outImg = None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


	cv2.namedWindow("output", cv2.WINDOW_NORMAL)     
	cv2.imshow('output',output)


	if cv2.waitKey(1000000000) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()









