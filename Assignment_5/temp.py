import cv2

cap = cv2.VideoCapture('Downloads/Corridor Original.mp4')

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
# cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
# ret, old_frame = cap.read()
# print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
# print(frame_count)
i = 0
while i < frame_count:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, old_frame = cap.read()
    print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    i = i+1