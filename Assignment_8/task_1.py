import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = 'Downloads/VO_Practice_Sequence/images'

class visual_odometry:
    def __init__(self,path) -> None:
        self.path = path
        self.files = os.listdir(path)
        self.cameraMatrix = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02],
                            [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02], 
                            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        self.C = np.hstack((np.eye(3),np.zeros((3,1))))
        self.pathX = []
        self.pathY = []

    def match_features_of_two_images(self, img1, img2):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        # good = []
        pts1 = []
        pts2 = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                # good.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        # # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(img3),plt.show()
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        return pts1, pts2
    
    def find_current_position(self, pts1, pts2):
        E,_ = cv2.findEssentialMat(pts1, pts2, self.cameraMatrix)
        _, R, T, _ = cv2.recoverPose(E, pts1,pts2, self.cameraMatrix)

        tK = np.hstack((R,T))
        tK = np.vstack((tK, np.array([[0, 0, 0, 1]])))

        self.C = self.C @ tK
        curTranslation = self.C[:,3].reshape(3)
        x,y = curTranslation[2],curTranslation[0]
        return x,y

    def update(self):
        for i in range(len(self.files)-1):
            filename1 = self.path + '/' + self.files[i]
            img1 = cv2.imread(filename1,0)            
            filename2 = self.path + '/' + self.files[i+1]
            img2 = cv2.imread(filename2,0)
            pts1, pts2 = self.match_features_of_two_images(img1,img2)
            x,y = self.find_current_position(pts1,pts2)
            self.pathX.append(x)
            self.pathY.append(y)
            print(i)
        
        plt.plot(self.pathX, self.pathY)
        plt.show()


VO = visual_odometry(path)
VO.update()

        







