import os
import cv2 #pip install opencv-python

video_dir = 'YouTubeClips'

def decompose_video(frame_path, video_input, filetype='png', skip=15):
    video_path = os.path.join(video_dir, video_input)
    capture = cv2.VideoCapture(video_path)
    frame_number = 0
    while True:
        success, frame = capture.read()
        if frame_number % skip == 0:
            if success:
                filename = os.path.join(frame_path, str(frame_number)+'.'+filetype)
                cv2.imwrite(filename, frame)
            else:
                break
        frame_number += 1
    capture.release()
    return