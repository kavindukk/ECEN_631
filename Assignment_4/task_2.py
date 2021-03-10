#!/usr/bin/env python3
import cv2
import numpy as np
import imutils
from numpy.core.defchararray import count
from numpy.lib.twodim_base import mask_indices

cap_L = cv2.VideoCapture('Downloads/Baseball Trajectory/BaseBall_Pitch_L.avi')
cap_R = cv2.VideoCapture('Downloads/Baseball Trajectory/BaseBall_Pitch_R.avi')

ret_L, frame_L = cap_L.read()
ret_L, frame_R = cap_L.read()

centerL = (355,98)
centerR = (272,99)
extension = 68

startL = (centerL[0] - extension, centerL[1] - extension)
endL = (centerL[0] + extension, centerL[1] + extension)

startR = (centerR[0] - extension, centerR[1] - extension)
endR = (centerR[0] + extension, centerR[1] + extension)
count = 1
# color = (255, 0, 0)
# thickness = 2
# count = 1

def process_the_image(img, center, extension):
	margin = 30
	extension = extension + margin
	crop = img[center[1] - extension : center[1] + extension, center[0] - extension : center[0] + extension]
	gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)	
	mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY )[1]
	maskedimg = cv2.bitwise_and(gray, gray, mask=mask)	
	detected_circles = cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 25, minRadius = 1, maxRadius = 40)
	if detected_circles is not None:
		detected_circles = list(detected_circles[0])
		detected_circles.sort(key=lambda x: x[2])
		print('circle:', detected_circles)
		return detected_circles[-1]
	print()
	return []

def draw_contours(img, contour, center, extension):
	margin = 30
	extension = extension + margin			
	if len(contour) > 0:	
		a,b,r = contour	
		a = int(center[0]-extension + a)
		b = int(center[1]-extension + b)  
		cv2.circle(img, (a, b), int(r), (0, 255, 0), 2)   
        # cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
	return img

while True:
	count = count + 1
	retL, frameL = cap_L.read()
	retR, frameR = cap_R.read()

	cv2.rectangle(frameL, startL, endL, (255,0,0), 1)
	cv2.rectangle(frameR, startR, endR, (255,0,0), 1)

	contoursL = process_the_image(frameL, centerL, extension)
	contoursR = process_the_image(frameR, centerR, extension)

	frameL = draw_contours(frameL, contoursL, centerL, extension)
	frameR = draw_contours(frameR, contoursR, centerR, extension)
	imgFinal = cv2.hconcat([frameL, frameR])
	cv2.imshow('Ball Detector', imgFinal)
	k = cv2.waitKey(0) & 0xFF
	if k == 27:
		cv2.destroyAllWindows()
		break
	if k == 87:
		cv2.imwrite('Ball detector'+str(count), imgFinal)
cv2.destroyAllWindows() # close the window      