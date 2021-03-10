import numpy as np
import cv2

camMatrixL = np.load('Parameters/Left_Cali/intrinsic_parameters.npy')
camMatrixR = np.load('Parameters/Right_Cali/intrinsic_parameters.npy')
distCoeffsL = np.load('Parameters/Left_Cali/distortion_parameters.npy')
distCoeffsR = np.load('Parameters/Right_Cali/distortion_parameters.npy')

R1 = np.load('R1_3x3_rectification_transform.npy')
R2 = np.load('R2_3x3_rectification_transform.npy')
P1 = np.load('P1_3x4_projection_matrix.npy')
P2 = np.load('P2_3x4_projection_matrix.npy')
Q = np.load('Q_4x4_disparity_to_depth_mapping_matrix.npy')

def find_corners(imagePath):
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((7*10,3), np.float32)
	objp[:,:2] = 3.88*np.mgrid[0:10,0:7].T.reshape(-1,2)

	objpoints = []
	imgpoints = [] 

	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (7,10), None)
	if ret == True:
		objpoints.append(objp)
		corners = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
	return corners

cornersL = find_corners('Downloads/stereo/L5.png')
cornersR = find_corners('Downloads/stereo/R5.png')

imageL = cv2.imread('Downloads/stereo/L5.png')
imageR = cv2.imread('Downloads/stereo/R5.png')

fourPointsL = np.array([cornersL[0], cornersL[6], cornersL[63],cornersL[69]])
fourPointsR = np.array([cornersR[0], cornersR[6], cornersR[63],cornersR[69]])

imageL = cv2.drawChessboardCorners(imageL, (2,2), fourPointsL, True)
imageR = cv2.drawChessboardCorners(imageR, (2,2), fourPointsR, True)

combine_image = np.hstack((imageL,imageR))
cv2.imshow('combine_image_L_and_R.png', combine_image)
key = cv2.waitKey(0) 

print('fourPoints_L',fourPointsL)
undistortPoints_L = cv2.undistortPoints(fourPointsL, camMatrixL, distCoeffsL, R=R1, P=P1)
undistortPoints_R = cv2.undistortPoints(fourPointsR, camMatrixR, distCoeffsR, R=R2, P=P2)

disparity = np.zeros([1,4])
disparity[0,0] = undistortPoints_L.item(0) - undistortPoints_R.item(0)
disparity[0,1] = undistortPoints_L.item(2) - undistortPoints_R.item(2)
disparity[0,2] = undistortPoints_L.item(4) - undistortPoints_R.item(4)
disparity[0,3] = undistortPoints_L.item(6) - undistortPoints_R.item(6)

print('disparity = ', disparity)

point_3d_L = np.zeros((4,1,3))
for i in range(4):
	point_3d_L[i, 0, 2] = disparity[0, i]
	for j in range(2):
		point_3d_L[i, 0, j] = undistortPoints_L[i, 0, j]

point_3d_R = np.zeros((4,1,3))
for i in range(4):
	point_3d_R[i, 0, 2] = disparity[0, i]
	for j in range(2):
		point_3d_R[i, 0, j] = undistortPoints_R[i, 0, j]



print('point_3d_L = ', point_3d_L)
print('point_3d_R = ', point_3d_R)

perspectiveTransform_L = cv2.perspectiveTransform(point_3d_L, Q)
print('perspectiveTransform_L = ', perspectiveTransform_L)

perspectiveTransform_R = cv2.perspectiveTransform(point_3d_R, Q)
print('perspectiveTransform_R = ', perspectiveTransform_R)

cv2.destroyAllWindows()