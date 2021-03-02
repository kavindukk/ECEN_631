import numpy as np
import cv2
import os 

def callibrate_camera(imgCategory, cameraName):    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)    
    objp = np.zeros((7*10,3), np.float32)
    objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    imagesFolder = os.path.join(os.getcwd(),'Downloads',str(imgCategory), str(cameraName))
    images = [ os.path.join(imagesFolder, f) for f in os.listdir('Downloads/'+ str(imgCategory) +'/'+str(cameraName)) if f.endswith(".png") or f.endswith('.bmp')]

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (10,7),None)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(50)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print('Intrinsic  parameters from '+ str(cameraName) + '= \n',mtx)
    print('Distortion parameters from '+ str(cameraName) + '= \n',dist,'\n')

    np.save('Parameters/'+str(imgCategory)+'/'+ str(cameraName) +'/'+'intrinsic_parameters.npy', mtx)
    np.save('Parameters/'+str(imgCategory)+'/'+ str(cameraName) +'/'+'distortion_parameters.npy', dist)

    cv2.destroyAllWindows()

##Practice Images
callibrate_camera('Practice_Images', 'L')
callibrate_camera('Practice_Images', 'R')

##Test Images
#Left Camera Callibration
callibrate_camera('Test_Images','Left_Cali')
#Right Camera Callibration
callibrate_camera('Test_Images','Right_Cali')



