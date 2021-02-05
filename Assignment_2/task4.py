import cv2
import numpy as np
import os

# Load previously saved data base on task2
mtx = np.load('intrinsic  parameters.npy')
dist = np.load('distortion parameters.npy')

objpoints = [] # 3d point in 3D world 
imgpoints = [] # 2d points in image plane.

# Get picture and Data Points
imagePath = os.path.join(os.getcwd(),'images','Object with Corners.jpg')
image = cv2.imread(imagePath)
data_points = open('datapoints.txt')

for line in data_points:   # deal with data line by line
    data = np.array(line.split())    # split data of each line
    if len(data) == 2:     # image points
        imgpoints.append(data)
    elif len(data) == 3:   # 3D object points
        objpoints.append(data)
    else:
        exit()

imgpoints = np.asarray(imgpoints, dtype=np.float32)
objpoints = np.asarray(objpoints, dtype=np.float32)

rvecs, tvecs = cv2.solvePnP(objpoints, imgpoints, mtx, dist)[1:3]

print('Rotation Vector = ', rvecs)
print('Translation Vector = ', tvecs)

# Find rotation and translation matrices.
rmat = cv2.Rodrigues(rvecs)[0]
tmat = cv2.Rodrigues(tvecs)[0]

print('Rotation Matrix = ', rmat)
print('Translation Matrix = ', tmat)

cv2.destroyAllWindows()