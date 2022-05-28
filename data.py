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
from tqdm import tqdm #pip install tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class Dataset():
    def __init__(self, num_examples=20, vid_dir = "YouTubeClips/",
                 img_dir="all_frames/", skip=15, keep=False):
        vid2tex, tex2vid = get_dictionaries()
        if not keep:
            breakdown_video(vid2tex, tex2vid, vid_dir, img_dir, num_examples, skip)
        self.image_names = os.listdir(img_dir)
        self.text_names = [name.split("_")[0] for name in self.image_names]
        self.images = torch.cat( #Create a tensor representation for the images
            [preprocess(Image.open(os.path.join(img_dir, img))).unsqueeze(0).to(device)
            for img in self.image_names]
        ).to(device)
        self.texts = torch.cat(
            [clip.tokenize(input) for input in self.text_names]
        ).to(device)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]

    def __len__(self):
        return self.images.shape[0]

    def get_names_lists(self):
        return self.image_names, self.text_names

def get_dictionaries(vid2tex_filename="vid2tex.json",
                    tex2vid_filename="tex2vid.json"):
    if not os.path.isfile(vid2tex_filename) or not os.path.isfile(tex2vid_filename):
        dataset_text_parser.export_descriptions()
    with open(vid2tex_filename) as f1, open(tex2vid_filename) as f2:
        vid2tex = json.load(f1)
        tex2vid = json.load(f2)
    return vid2tex, tex2vid

def breakdown_video(vid2tex, tex2vid, vid_dir, img_dir, num_examples, skip,
                    vid_type=".avi", img_type=".png"):
    if os.path.exists(img_dir):
        print(f"Deleting old frames saved in {img_dir}")
        for root, dirs, old_files in os.walk(img_dir):
            for old in old_files:
                os.remove(os.path.join(root, old))
    else:
        print(f"Creating new folder {img_dir}")
        os.mkdir(img_dir)
    print(f"Saving new frames in {img_dir}\n")
    for vid, tex in list(vid2tex.items())[:num_examples]:
        vid_path = os.path.join(vid_dir, vid + vid_type)
        capture = cv2.VideoCapture(vid_path)
        frame_num = 0
        while True:
            success, frame = capture.read()
            if frame_num % skip == 0:
                if success:
                    filename = os.path.join(
                        img_dir, tex[0] + "_" + str(frame_num) + img_type
                    )
                    if os.path.exists(filename):
                        os.remove(filename)
                    cv2.imwrite(filename, frame)
                else:
                    break
            frame_num += 1
    capture.release()
