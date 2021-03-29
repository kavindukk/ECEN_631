import numpy as np
import cv2 

cimg1 = cv2.imread('Downloads/im1.jpg')          
cimg2 = cv2.imread('Downloads/b1.png') 
img1 = cv2.imread('Downloads/im1.jpg',cv2.IMREAD_GRAYSCALE)          
img2 = cv2.imread('Downloads/b1.png',cv2.IMREAD_GRAYSCALE) 

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(cimg1,kp1,cimg2,kp2,matches[:50],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Task1_Matching_Results', img3)
cv2.imwrite('Task1_Matching_Result.jpg',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()