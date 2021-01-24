import cv2
import numpy as np
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
vout = cv2.VideoWriter('./video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))

functionNo = 2
key = cv2.waitKey(1) & 0xFF
while True:
    if functionNo == 1:
        ret0, frame = camera.read()        
        cv2.imshow("Original_Camera", frame)
        vout.write(frame)
        key = cv2.waitKey(1) & 0xFF

    elif functionNo == 2:
        ret0, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresholding_function", threshold)
        vout.write(threshold)
        key = cv2.waitKey(1) & 0xFF

    elif functionNo == 3:
        ret0, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,100,200)
        cv2.imshow("Edges_function", edges)
        vout.write(edges)
        key = cv2.waitKey(1) & 0xFF

    elif functionNo == 4:
        ret0, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        frame[dst>0.01*dst.max()]=[0,0,255]
        cv2.imshow("Canny_Corner_function", dst)
        vout.write(dst)
        key = cv2.waitKey(1) & 0xFF

    elif functionNo == 5:
        ret0, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        # lines = cv2.HoughLines(edges,1,np.pi/180,200)
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Hough_Lines_function", frame)
        vout.write(frame)
        key = cv2.waitKey(1) & 0xFF

    elif functionNo == 6:
        ret,frame1 = camera.read()
        gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        ret,frame2 = camera.read()
        gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1,gray2)
        cv2.imshow('Differet Detection',diff)
        vout.write(diff)
        key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
        cap.release()
    elif key == ord('1'):
        functionNo = 1
        cv2.destroyAllWindows()
    elif key == ord('2'):
        functionNo = 2
        cv2.destroyAllWindows()
    elif key == ord('3'):
        functionNo = 3
        cv2.destroyAllWindows()
    elif key == ord('4'):
        functionNo = 4
        cv2.destroyAllWindows()
    elif key == ord('5'):
        functionNo = 5
        cv2.destroyAllWindows()
    elif key == ord('6'):
        functionNo = 6
        cv2.destroyAllWindows()

camera.release()
cv2.destroyAllWindows()