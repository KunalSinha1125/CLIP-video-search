import os
import cv2 #pip install opencv-python

video_dir = 'YouTubeClips'

def decompose_video(frame_path, video_input, skip, filetype='png'):
    video_path = os.path.join(video_dir, video_input)
    capture = cv2.VideoCapture(video_path)
    frame_number = 0
    images_list = []
    while True:
        success, frame = capture.read()
        if frame_number % skip == 0:
            if success:
                filename = os.path.join(frame_path, str(frame_number)+'.'+filetype)
                cv2.imwrite(filename, frame)
                images_list.append(filename)
            else:
                break
        frame_number += 1
    capture.release()
    return images_list
