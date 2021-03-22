import numpy as np
import cv2

cap = cv2.VideoCapture('Downloads/Corridor Original.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

ret, frame = cap.read()
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_points = cv2.goodFeaturesToTrack(frame_gray,
                                mask=None,
                                maxCorners=50,
                                qualityLevel=0.1,
                                minDistance=10,
                                blockSize=7)

while int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < frame_count:
    ret, new_frame = cap.read()
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_points = np.zeros_like(frame_points)

    for i, point in enumerate(frame_points):
        point = point[0]
        search_window = new_gray[int(point[1]):int(point[1]+50), int(point[0]):int(point[0]+50)]
        template_window = frame_gray[int(point[1]):int(point[1]+10), int(point[0]):int(point[0]+10)]
        resmat = cv2.matchTemplate(search_window, template_window, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resmat)
        point = [int(point[0])+max_loc[0],int(point[1])+max_loc[1]]
        new_points[i] = point

    frame_points = np.int32(frame_points)
    new_points = np.int32(new_points)
    F, mask = cv2.findFundamentalMat(frame_points, new_points, cv2.FM_RANSAC)
    frame_points = frame_points[mask.ravel() == 1]
    new_points = new_points[mask.ravel() == 1]

    for i,(new,frame) in enumerate(zip(new_points,frame_points)):
        a,b = new.ravel()
        c,d = frame.ravel()  
        new_frame = cv2.line(new_frame, (a,b),(c,d), (0,0,255), 2)
        new_frame = cv2.circle(new_frame,(a,b),3,(0,255,0),3)

    frame_points = new_points
    frame_gray = new_gray

    new_frame = cv2.putText(new_frame, 'Skipped Frames:0', (400, 700), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.imshow('frame',new_frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()