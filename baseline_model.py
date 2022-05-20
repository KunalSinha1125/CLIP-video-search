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

frame_dir = 'frames/'
video_dir = 'YouTubeClips/'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main(num_examples, top_k, skip):
    vid2tex, tex2vid = get_dictionaries()
    image_inputs = get_images(num_examples, skip)
    text_inputs = get_texts(num_examples, vid2tex)
    similarity = compute_similarity(image_inputs, text_inputs)
    analyze_results(similarity, num_examples, image_inputs,
                    text_inputs, vid2tex, tex2vid, top_k)

def get_dictionaries(vid2tex_filename="vid2tex.json",
                    tex2vid_filename="tex2vid.json"):
    if not os.path.isfile(vid2tex_filename) or not os.path.isfile(tex2vid_filename):
        dataset_text_parser.export_descriptions()
    with open(vid2tex_filename) as f1, open(tex2vid_filename) as f2:
        vid2tex = json.load(f1)
        tex2vid = json.load(f2)
    return vid2tex, tex2vid

def get_images(num_examples=5, skip=15):
    video_list = os.listdir(video_dir)
    image_inputs = []
    for i in range(num_examples):
        video_input = video_list[i]
        frame_path = os.path.join(frame_dir, video_input)
        if os.path.exists(frame_path):
            images_list = [os.path.join(frame_path, image)
                for image in os.listdir(frame_path)]
        else:
            print(f"Saving frames for video {video_input}")
            os.makedirs(frame_path)
            #images_list = dataset_processor.decompose_video(frame_path, video_input, skip)
            images_list = keyframe.decompose_video1(frame_path, video_input)
        for image in images_list:
            image_inputs.append(image)
    return image_inputs

def get_texts(num_examples, vid2tex):
    text_inputs = [desc[0] for desc in vid2tex.values()]
    return text_inputs[:num_examples]

def compute_similarity(image_inputs, text_inputs):
    images = torch.cat( #Create a tensor representation for the images
        [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_inputs]
    ).to(device)
    texts = torch.cat(
        [clip.tokenize(input) for input in text_inputs]
    ).to(device)
    _, similarity = model(images, texts)
    return similarity

def analyze_results(similarity, num_examples, image_inputs, text_inputs, vid2tex, tex2vid, top_k=1):
    accuracy = 0
    values, indices = similarity.topk(top_k)
    for i, (value, index) in enumerate(zip(values, indices)):
        actual = tex2vid[text_inputs[i]]
        pred = image_inputs[int(index)]
        pred = pred[pred.index("/")+1 : pred.index(".")]
        accuracy += (pred == actual[0])
        print(f"\nText input: {text_inputs[i]}")
        print(f"Model prediction: video {pred}. Depicts: {vid2tex[pred][0]}")
        print(f"Frame: {image_inputs[index]:>16s}")
        print(f"Probability: {value.item():.2f}%")
    accuracy /= num_examples
    print(f"\nOverall Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_examples',
                        default=20,
                        help='How many examples to test on?')
    parser.add_argument('--top_k',
                        default=1,
                        help='How many frames to retrieve?')
    parser.add_argument('--skip',
                        default=15,
                        help='How many frames to skip while saving?')
    args = parser.parse_args()
    main(int(args.num_examples), int(args.top_k), int(args.skip))


'''
def get_image_embeddings(image_inputs):
    images = torch.cat( #Create a tensor representation for the images
        [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_inputs]
    ).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images) #Produce image embeddings
    return image_features

def get_text_embedding(text_input):
    text = clip.tokenize(text_input).to(device) #Create a tensor representation for the text
    with torch.no_grad():
        text_features = model.encode_text(text) #Produce text embeddings
    return text_features

def run_clip(vid2tex, tex2vid, image_inputs, text_input,
                image_features, text_features, top_k=1):
    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(top_k)
    # Print the result
    print(f"\nUser input: {text_input}:")
    return num_correct
'''
