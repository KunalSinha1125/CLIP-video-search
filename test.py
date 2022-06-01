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
import fine_tune
import data
from tqdm import tqdm #pip install tqdm
import h_clustering

frame_dir = 'all_frames/'
video_dir = 'YouTubeClips/'

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(num_examples, top_k, save_fps, keep, frame_type, model_type, batch_size, print_preds, threshold):
    model, preprocess = clip.load("ViT-B/32", device=device)
    if model_type == "finetuned":
        model = fine_tune.load()
    test_dataset = data.Dataset(
        num_examples=num_examples, save_fps=save_fps, keep=keep
    )
    if frame_type == "keyframe":
        keyframes = h_clustering.clusterKeyFrames(test_dataset, batch_size, threshold)
        test_dataset.delete_redundant_frames(keyframes)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    num_correct = 0
    for i, (images, texts, image_names, text_names) in enumerate(tqdm(test_dataloader, desc="Testing batches")):
        #We process the images in batches but feed in all unique text descriptions at once
        texts = test_dataset.unique_texts
        _, similarity = model(images, texts)
        num_correct += analyze_results(
            test_dataset, similarity, num_examples, images, texts, top_k, print_preds
        )
    print(f"\nFinal accuracy is {num_correct / len(test_dataset.unique_text_names)}")

def analyze_results(test_dataset, similarity, num_examples, image_inputs, text_inputs, top_k=1, print_preds=False):
    num_correct = 0
    values, indices = similarity.topk(top_k)
    for i, (value, index) in enumerate(zip(values, indices)):
        actual = test_dataset.unique_text_names[i]
        pred = test_dataset.text_names[index]
        num_correct += (actual == pred)
        if print_preds:
            print(f"Actual: {actual}")
            print(f"Predicted: video {pred}")
            print(f"Probability: {value.item():.2f}%")
    return num_correct

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_examples',
                        default=20,
                        help='How many examples to test on?')
    parser.add_argument('--top_k',
                        default=1,
                        help='How many frames to retrieve?')
    parser.add_argument('--save_fps',
                        default=1,
                        help='How many frames per second to save?')
    parser.add_argument('--keep',
                        action='store_true',
                        #default = 'keep',
                        help='Specify whether to re-save the frames')
    parser.add_argument('--frame_type',
                        default='keyframe',
                        choices=['baseline', 'keyframe'],
                        help='Specify how to save the frames')
    parser.add_argument('--model_type',
                        default='baseline',
                        choices=['baseline', 'finetuned'],
                        help='Specify which model to use')
    parser.add_argument('--batch_size',
                        default=32,
                        help='What should the batch size be?')
    parser.add_argument('--print_preds',
                        action='store_true',
                        help='''Should we print each model prediction?
                            Warning: could take up a lot of space''')
    parser.add_argument('--threshold',
                        default=0.05,
                        help='Set distance threshold for clustering')
    args = parser.parse_args()
    main(int(args.num_examples), int(args.top_k), int(args.save_fps), args.keep,
         args.frame_type, args.model_type, int(args.batch_size), args.print_preds,
         float(args.threshold))
