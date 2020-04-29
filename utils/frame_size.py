#check frame size
import cv2
import numpy as np
INPUT_STREAM = "edit_test.mp4" #shape of frame is (720, 1280, 3)
cap = cv2.VideoCapture(INPUT_STREAM)
while cap.isOpened():
    flag, frame = cap.read()
    frame_arr = np.array(frame)
    print('shape of frame is {}'.format(frame_arr.shape))
    break
    
