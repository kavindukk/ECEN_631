import numpy as np
import cv2
import os

mtx = np.load('intrinsic  parameters.npy')
dist = np.load('distortion parameters.npy')

def input_path(name):
    inputPath = os.path.join(os.getcwd(), 'images', str(name+'.jpg'))
    input_path = str(inputPath)
    return input_path

def output_path(name):
    outputPath = os.path.join(os.getcwd(), 'images', 'results', str('task3_'+name+'.jpg'))
    return str(outputPath)

def different(name):
    imagePath = input_path(name)
    img = cv2.imread(imagePath)  
    h, w = img.shape[:2]
    h += 1
    w += 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # undistort
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

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

different('Close')
different('Far')
different('Turned')

cv2.waitKey(0)
cv2.destroyAllWindows()