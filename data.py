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
FPS = 30

class Dataset():
    def __init__(self, num_examples=20, vid_dir = "YouTubeClips/", data_type="baseline",
                 img_dir="all_frames/", save_fps=1, keep=False):
        vid2tex, tex2vid = get_dictionaries()
        self.frames_saved = 0
        if keep:
            self.frames_saved = len(os.listdir(img_dir))
        else:
            self.frames_saved = breakdown_video(
                vid2tex, tex2vid, vid_dir, img_dir, num_examples, save_fps, data_type
            )
        self.image_names = np.array(os.listdir(img_dir))
        self.images = torch.cat( #Create a tensor representation for the images
            [preprocess(Image.open(os.path.join(img_dir, img))).unsqueeze(0).to(device)
            for img in self.image_names]
        ).to(device)
        self.text_names = np.array([name.split("_")[0] for name in self.image_names])
        self.texts = torch.cat(
            [clip.tokenize(input) for input in self.text_names]
        ).to(device)
        self.unique_text_names = list(dict.fromkeys(self.text_names))
        self.unique_texts = torch.cat(
            [clip.tokenize(input) for input in self.unique_text_names]
        ).to(device)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]

    def __len__(self):
        return self.images.shape[0]

    def get_num_frames_saved(self):
        return self.frames_saved

    def delete_redundant_frames(self, keyframes):
        print(f"Keyframes list: {keyframes}")
        mask = np.zeros(self.images.shape[0], dtype=bool)
        mask[keyframes] = True
        self.images = self.images[mask]
        self.image_names = self.image_names[mask]
        self.texts = self.texts[mask]
        self.text_names = self.text_names[mask]

def get_dictionaries(vid2tex_filename="vid2tex.json",
                    tex2vid_filename="tex2vid.json"):
    if not os.path.isfile(vid2tex_filename) or not os.path.isfile(tex2vid_filename):
        dataset_text_parser.export_descriptions()
    with open(vid2tex_filename) as f1, open(tex2vid_filename) as f2:
        vid2tex = json.load(f1)
        tex2vid = json.load(f2)
    return vid2tex, tex2vid

def breakdown_video(vid2tex, tex2vid, vid_dir, img_dir, num_examples, save_fps,
                    data_type, vid_type=".avi", img_type=".png"):
    if os.path.exists(img_dir):
        print(f"Deleting old frames saved in {img_dir}")
        for root, dirs, old_files in os.walk(img_dir):
            for old in old_files:
                os.remove(os.path.join(root, old))
    else:
        print(f"Creating new folder {img_dir}")
        os.mkdir(img_dir)
    print(f"Saving new frames in {img_dir}")
    data_list = list(vid2tex.items())
    if data_type == "finetuned":
        data_list = data_list[::-1]
    frames_saved = 0
    for vid, tex in data_list[:num_examples]:
        frame_num = 0
        vid_path = os.path.join(vid_dir, vid + vid_type)
        capture = cv2.VideoCapture(vid_path)
        while True:
            success, frame = capture.read()
            if frame_num % (FPS / save_fps) == 0:
                if success:
                    filename = os.path.join(
                        img_dir, tex[0] + "_" + str(frame_num) + img_type
                    )
                    if os.path.exists(filename):
                        os.remove(filename)
                    cv2.imwrite(filename, frame)
                    frames_saved += 1
                else:
                    break
            frame_num += 1
    capture.release()
    return frames_saved
