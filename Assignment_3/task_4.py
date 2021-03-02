import cv2
import numpy as np

L_intrinsic = np.load('Parameters/Test_images/Left_Cali/intrinsic_parameters.npy')
R_intrinsic = np.load('Parameters/Test_images/Left_Cali/intrinsic_parameters.npy')
L_distortion = np.load('Parameters/Test_images/Right_Cali/distortion_parameters.npy')
R_distortion = np.load('Parameters/Test_images/Right_Cali/distortion_parameters.npy')
cameraParameters = [ L_intrinsic, L_distortion, R_intrinsic, R_distortion]

rotation_matrix = np.load('rotation_matrix.npy')
translation_vector = np. load('translation_vector.npy')
pose = [ rotation_matrix, translation_vector]

def compute_stereo_rectification_maps(imgLName, imgRName, camParams, PoseParams):
    imgL = cv2.imread(imgLName)
    imgR = cv2.imread(imgRName)
    h, w = imgL.shape[:2]
    imgSize = (w, h)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camParams[0], camParams[1], camParams[2], camParams[3],
                                                      imgSize, PoseParams[0], PoseParams[1])
    map1x, map1y = cv2.initUndistortRectifyMap(camParams[0], camParams[1], R1, P1, imgSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(camParams[2], camParams[3], R2, P2, imgSize, cv2.CV_32FC1)

    remapL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    remapY = cv2.remap(imgL, map2x, map2y, cv2.INTER_LINEAR)
    return remapL, remapY

def Difference(imagename_ori, remap):
    img_ori = cv2.imread(imagename_ori)
    gray1 = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(remap, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff

imgL = 'Downloads/Test_Images/Stereo_Left/L1.png'
imgR = 'Downloads/Test_Images/Stereo_Right/R1.png'

remapL, remapR = compute_stereo_rectification_maps(imgL, imgR, cameraParameters, pose)
diffL = Difference(imgL, remapL)
diffR = Difference(imgR, remapR)

diffLC = cv2.cvtColor(diffL, cv2.COLOR_GRAY2BGR)
diffRC = cv2.cvtColor(diffR, cv2.COLOR_GRAY2BGR)

for y in range(20):
    cv2.line(remapL, (0, y*32), (640, y*32), (0, 0, 255), 1)
    cv2.line(remapR, (0, y*32), (640, y*32), (0, 0, 255), 1)
hor1 = cv2.hconcat([remapL, remapR])
hor2 = cv2.hconcat([diffLC, diffRC])
finalImg = cv2.vconcat([hor1, hor2])
cv2.imshow('Task_4', finalImg)
cv2.waitKey(0)
cv2.destroyAllWindows()