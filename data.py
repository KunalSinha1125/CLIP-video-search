import os
import torch
import clip #pip install clip
from PIL import Image, ImageSequence
import cv2 #pip install opencv-python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import dataset_text_parser
import dataset_processor
import json
from time import sleep
from progress.bar import Bar #pip install progress
import keyframe
from baseline_model import frame_dir, video_dir

def get_images(num_examples=5, skip=15, save=True, model_type="baseline"):
    video_list = os.listdir(video_dir)
    image_inputs = []
    for i in range(num_examples):
        video_input = video_list[i]
        frame_path = os.path.join(frame_dir, video_input)
        if save:
            if os.path.exists(frame_path):
                print(f"Deleting old frames saved in {video_input}")
                for root, dirs, old_files in os.walk(frame_path):
                    for old in old_files:
                        os.remove(os.path.join(root, old))
            else:
                os.makedirs(frame_path)
            print(f"Saving new frames for video {video_input}")
            if model_type == "baseline":
                images_list = dataset_processor.decompose_video(frame_path, video_input, skip)
            elif model_type == "keyframe":
                images_list = keyframe.decompose_video1(frame_path, video_input)
        else: #os.path.exists(frame_path):
            images_list = [os.path.join(frame_path, image)
                for image in os.listdir(frame_path)]
        for image in images_list:
            image_inputs.append(image)
    return image_inputs
