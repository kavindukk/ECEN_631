import numpy as np
import cv2

def calc_optical_flow_using_feature_matching(self, noOfSkippedFrames = 10):
    cap = cv2.VideoCapture('Downloads/Corridor Original.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    while frame_number < frame_count:
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        point0 = cv2.goodFeaturesToTrack(old_gray,
                                    mask=None,
                                    maxCorners=100,
                                    qualityLevel=0.1,
                                    minDistance=10,
                                    blockSize=7)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, new_frame = cap.read()
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_point = np.zeros_like(point0)

        for i, point in enumerate (point0):
            point = point[0]
            search_window = new_gray[int(point[1]):int(point[1]+50), int(point[0]):int(point[0]+50)]
            template_window = old_gray[int(point[1]):int(point[1]+10), int(point[0]):int(point[0]+10)]
            resmat = cv2.matchTemplate(search_window, template_window, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resmat)
            point1 = [int(point[0])+max_loc[0],int(point[1])+max_loc[1]]
            new_point[i] = point1

        for i,(new,old) in enumerate(zip(new_point,point0)):
            a,b = new.ravel()
            c,d = old.ravel()
            new_frame = cv2.line(new_frame, (a,b),(c,d), (0,0,255), 2)
            new_frame = cv2.circle(new_frame,(a,b),3,(0,255,0),3)

        new_frame = cv2.putText(new_frame, 'No of Skipped Frames:10', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('frame',new_frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        frame_number += noOfSkippedFrames
    cv2.destroyAllWindows()
    cap.release()