import os
import random
import cv2

video_file = "/nas/datasets/UCF-101/CliffDiving/v_CliffDiving_g01_c01.avi"
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("无法打开")
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for i in range(8):
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames * i // 8 - 1)
    success, image = cap.read()
    cv2.imwrite(f'./demo/{i}.jpg', image)
cap.release()
cv2.destroyAllWindows()