import numpy as np
import cv2 
import os


for i in range(5,40):
    if i <= 8:
        s1 = "0"+str(i)
        s2 = "0"+str(i+1)
    elif i == 9:
        s1 = "0"+str(i) 
        s2 = str(i+1)       
    else:
        s1 = str(i) 
        s2 = str(i+1)

    nameImg1 = "1L"+s1+".jpg"
    nameImg2 = "1L"+s2+".jpg"
    abs_path1 = os.path.join(os.getcwd(),'Assignment_1', 'Downloads', 'baseball', 'jpg', nameImg1)
    abs_path2 = os.path.join(os.getcwd(),'Assignment_1', 'Downloads', 'baseball', 'jpg', nameImg2)
    img1 = cv2.imread(abs_path1)
    img2 = cv2.imread(abs_path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image_sub = cv2.absdiff(gray1, gray2)

    ret,thres = cv2.threshold(image_sub,10,255,cv2.THRESH_BINARY)	
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thres,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    circles = cv2.HoughCircles(dilation, cv2.HOUGH_GRADIENT, 1,1000, 1,100,1,0,25)
    if circles.shape[0] !=4:
        x, y, r = circles[0,0] 
        cv2.circle(img2, (x, y), int(r), (0, 255, 0), 3)
    cv2.imshow('output',img2)
    cv2.waitKey(50)

