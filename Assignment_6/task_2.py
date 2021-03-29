import numpy as np
import cv2
from numpy.lib.type_check import imag


reference = cv2.imread('Downloads/cropped.jpg', 0) 
target = cv2.imread('Downloads/im1.jpg')

width = reference.shape[1]
hight = reference.shape[0]
target_resized = cv2.resize(target,(width,hight),interpolation = cv2.INTER_AREA)

# cap = cv2.VideoCapture(0)
sift = cv2.SIFT_create()
kp_image, des_image = sift.detectAndCompute(reference,None)
# reference = cv2.drawKeypoints(reference, kp_image, None)

# Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# while True:
# ret, frame = cap.read()
frame = cv2.imread('Downloads/redu.jpg',0) 
cframe = cv2.imread('Downloads/redu.jpg') 

grayframe = frame
kp_grayframe, des_grayframe = sift.detectAndCompute(grayframe,None)

matches = flann.knnMatch(des_image,des_grayframe, k=2)
good_points = []
for m, n in matches:
    if m.distance < 0.5*n.distance:         
        good_points.append(m)

matches_result = cv2.drawMatches(reference,kp_image,frame,kp_grayframe,good_points,None, flags=2)

if len(good_points) > 10:
    image_points = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
    frame_points = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(image_points, frame_points, cv2.RANSAC, 5.0)

    h, w = reference.shape
    points = np.float32([[0,0],[0,h-6],[w-6,h-6],[w-6,0]]).reshape(-1, 1, 2)
    distort_pts = cv2.perspectiveTransform(points, matrix)
    homography = cv2.polylines(frame, [np.int32(distort_pts)], True, (255, 0, 0), 3)

    warp_result = cv2.warpPerspective(target_resized, matrix, (homography.shape[1], homography.shape[0]))
    gray_warp = cv2.cvtColor(warp_result,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_warp, 3, 255, cv2.THRESH_BINARY_INV)

    back = cv2.bitwise_and(cframe, cframe, mask=mask)
    final = cv2.bitwise_or(back, warp_result)
    cv2.imshow("final", final)
else:
    print("not enough matches found")

cv2.imshow('matches_result', matches_result)

key = cv2.waitKey(0)
# if key == 27:
#     break
# elif key =='s':
#     cv2.imwrite('')

cv2.release()
cv2.destroyAllWindows()