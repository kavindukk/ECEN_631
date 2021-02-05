import cv2
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
vout = cv2.VideoWriter('task1.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))
while True:
    ret0, frame = camera.read()
    cv2.imshow("Cam 0", frame)
    vout.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
# cv2.imwrite('./Cam0.jpg', frame)
camera.release()
cv2.destroyAllWindows()