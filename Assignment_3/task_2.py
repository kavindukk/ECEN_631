import numpy as np
import cv2
import os

def calculate_extrinsic_parameters(imageSet, CameraL, CameraR):
    L_intrinsic = np.load(str(CameraL)+' intrinsic  parameters.npy') 
    L_distortion = np.load(str(CameraL)+' distortion parameters.npy')
    R_intrinsic = np.load(str(CameraR) +' intrinsic  parameters.npy')
    R_distortion = np.load(str(CameraR) +' distortion parameters.npy') 
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((7*10,3), np.float32)				
    objp[:,:2] = 3.88 * np.mgrid[0:10,0:7].T.reshape(-1,2)	

    objpoints = [] # 3d point in real world space
    L_imgpoints = [] # 2d points in image plane.
    R_imgpoints = [] # 2d points in image plane.

    head_list = [str(CameraL), str(CameraR)]
    for head in head_list:
        imagesFolder = os.path.join(os.getcwd(),'Downloads', str(imageSet), str(head))
        images = [ os.path.join(imagesFolder, f) for f in os.listdir('Downloads/'+str(imageSet)+'/'+str(head)) if f.endswith(".png") or f.endswith(".bmp")]
        print('head = ', head)
        for pic in images:
            img = cv2.imread(pic)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (10,7),None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                if head == str(CameraL):
                    objpoints.append(objp)
                    L_imgpoints.append(corners2)
                    print('L_imgpoints pic =', pic)
                elif head == str(CameraR):
                    R_imgpoints.append(corners2)
                    print('R_imgpoints pic =', pic)
        
                img = cv2.drawChessboardCorners(img, (10,7), corners2,ret)
                cv2.imshow('image',img)
                cv2.waitKey(50)

    # using stereoCalibrate to get Rotation, Translation, Essential, Fundation 
    # termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    termination_criteria_extrinsics = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 1e-6)



    rms_stereo, stereo_camera_matrix_l, stereo_dist_coeffs_l, stereo_camera_matrix_r, stereo_dist_coeffs_r, R, T, E, F = \
        cv2.stereoCalibrate(objpoints, L_imgpoints, R_imgpoints, L_intrinsic, L_distortion, R_intrinsic, R_distortion,  gray.shape[::-1], criteria=termination_criteria_extrinsics, flags=cv2.CALIB_FIX_INTRINSIC)

    print('R = ', R)
    print('T = ', T)
    print('E = ', E)
    print('F = ', F)
    
    if CameraL == "Stereo_Left" and CameraR == "Stereo_Right":
        np.save('rotation_matrix.npy', R)
        np.save('translation_vector.npy', T)
        np.save('essential_matrix.npy', E)
        np.save('fundamental_matrix.npy', F)


calculate_extrinsic_parameters("Test_Images", "Stereo_Left", "Stereo_Right")
# calculate_extrinsic_parameters("Practice_Images", "SL", "SR")dte