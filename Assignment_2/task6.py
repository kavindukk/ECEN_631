import numpy as np
import cv2
import os

# load intrinsic and distortion parameters
mtx = np.load('Real-Time intrinsic  parameters.npy')
dist = np.load('Real-Time distortion parameters.npy')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:7].T.reshape(-1,2)

objpoints = [] 
imgpoints = [] 

def input_path(no):
    inputPath = os.path.join(os.getcwd(), 'images', 'task5', str('img_'+no+'.jpg'))
    input_path = str(inputPath)
    return input_path

def output_path(name):
    outputPath = os.path.join(os.getcwd(), 'images', 'results', str('task6_'+name+'.jpg'))
    return str(outputPath)

def different(name):
    imagePath = input_path(name)
    img = cv2.imread(imagePath)  
    h, w = img.shape[:2]
    h += 1
    w += 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    output = output_path(name)
    cv2.imwrite(output, dst)

    # Compare Original one and Distortion one
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    fileName = str(name)+'diff'
    output_diff = output_path(fileName)
    cv2.imwrite(output_diff, diff)

    cv2.imshow(name + ' Differet Detection', diff)
    return

different('1')
different('2')
different('3')

cv2.waitKey(0)
cv2.destroyAllWindows()