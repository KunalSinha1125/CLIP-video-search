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
                 img_dir="all_frames/", skip=15, save=True):
        vid2tex, tex2vid = get_dictionaries()
        if save:
            breakdown_video(vid2tex, tex2vid, vid_dir, img_dir, num_examples, skip)
        image_names = os.listdir(img_dir)
        text_descriptions = [name.split("_")[0] for name in image_names]
        self.images = torch.cat( #Create a tensor representation for the images
            [preprocess(Image.open(os.path.join(img_dir, img))).unsqueeze(0).to(device)
            for img in image_names]
        ).to(device)
        self.texts = torch.cat(
            [clip.tokenize(input) for input in text_descriptions]
        ).to(device)
        '''
        self.labels = torch.zeros((len(self)), dtype=torch.long)
        for i in range(len(self)):
            vid = image_names[i].split("_")[0]
            self.labels[i] = (text_descriptions.index(vid))
        '''
        '''
        num_imgs = len(os.listdir(img_dir))
        img_matrix = np.zeros((num_imgs, resolution, resolution, 3))
        img_names = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        print("Preprocessing images...")
        for i in tqdm(range(num_imgs)):
            img_matrix[i] = cv2.resize(cv2.imread(img_names[i]), (resolution, resolution))
        '''
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx]#, self.labels[idx]

    def __len__(self):
        return self.images.shape[0]

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
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
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
