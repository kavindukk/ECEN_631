import numpy as np
import cv2

L_intrinsic = np.load('Parameters/Test_images/Left_Cali/intrinsic_parameters.npy')
R_intrinsic = np.load('Parameters/Test_images/Left_Cali/intrinsic_parameters.npy')
L_distortion = np.load('Parameters/Test_images/Right_Cali/distortion_parameters.npy')
R_distortion = np.load('Parameters/Test_images/Right_Cali/distortion_parameters.npy')
fundamental_matrix = np.load('fundamental_matrix.npy')

def undistortion(name, camera_matrix, dist_coeffs):
    img = cv2.imread(name)
    h, w = img.shape[:2]
    h += 1
    w += 1
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    # cv2.imwrite('Undistortion' + name, dst)
    return dst

def find_corners_of_an_image(image_name, chessboard_size):
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)    
    return corners.reshape(-1,2)

def drawPoints(img, pts, colors):
    for pt, color in zip(pts, colors):
        cv2.circle(img, tuple(pt), 5, color, -1)

def drawLines(img, lines, colors):
    _, c, _ = img.shape
    for r, color in zip(lines, colors):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), color, 1)


undstL = undistortion('Downloads/Test_Images/Stereo_Left/L1.png', L_intrinsic, L_distortion)
undstR = undistortion('Downloads/Test_Images/Stereo_Right/R1.png', R_intrinsic, R_distortion)

imgptsL = find_corners_of_an_image('Downloads/Test_Images/Stereo_Left/L1.png', (7,10))
imgptsR = find_corners_of_an_image('Downloads/Test_Images/Stereo_Right/R1.png', (7,10))

ptsL = np.array([imgptsL[0], imgptsL[1], imgptsL[2]])
ptsR = np.array([imgptsR[-1], imgptsR[-2], imgptsR[-3]])
drawPoints(undstL, ptsL, (0, 0, 255))
drawPoints(undstR, ptsR, (255, 0, 0))

epilinesR = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, fundamental_matrix)
epilinesR = epilinesR.reshape(-1, 3)
drawLines(undstL, epilinesR, (255, 0, 0))

epilinesL = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, fundamental_matrix)
epilinesL = epilinesL.reshape(-1, 3)
drawLines(undstR, epilinesL, (0, 0, 255))

img = cv2.hconcat([undstL, undstR])
cv2.imshow('Epipolar_lines',img)

cv2.waitKey(0)
cv2.destroyAllWindows()