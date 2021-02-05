import numpy as np
import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read() 
  
    # Convert the img to grayscale 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Apply edge detection method on the image 
    edges = cv2.Canny(gray,50,150) 
    line_image = np.copy(img) * 0 
    # This returns an array of r and theta values 
    lines = cv2.HoughLinesP(edges,1,np.pi/180, 15, np.array([]), 50, 20) 
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    key = cv2.waitKey(1) & 0xFF