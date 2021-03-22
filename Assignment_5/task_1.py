import numpy as np
import cv2

def calc_optical_flow_between_two_frames(m = 10, maxLevel = 4):
    cap = cv2.VideoCapture('Downloads/Corridor Original.mp4')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count = ', frame_count)
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
        point_m, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                            new_gray,
                                            point0,
                                            None,
                                            winSize=(15,15),
                                            maxLevel= maxLevel,
                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        good_new = point_m[st==1]
        good_old = point0[st==1]

    
        for i,(new,old) in enumerate(zip(good_new,good_old)):

            a,b = new.ravel()
            c,d = old.ravel()

            new_frame = cv2.line(new_frame, (a,b),(c,d), (0,0,255), 2)
            new_frame = cv2.circle(new_frame,(a,b),3,(0,255,0),3)

        new_frame = cv2.putText(new_frame, 'Pyramid Level:4  Skipped Frames:10', (400, 700), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('frame',new_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        frame_number += m

    cv2.destroyAllWindows()
    cap.release()

calc_optical_flow_between_two_frames()