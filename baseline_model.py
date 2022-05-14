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

frame_dir = 'frames/'
video_dir = 'YouTubeClips/'
device = "cuda" if torch.cuda.is_available() else "cpu"
vid2tex_filename = "vid2tex.json"
tex2vid_filename = "tex2vid.json"
model, preprocess = clip.load("ViT-B/32", device=device)

def main(video_input, text_input, top_k, skip):
    frame_path = os.path.join(frame_dir, video_input)
    image_inputs = []
    if not os.path.exists(frame_path):
        print('Saving frames...')
        os.makedirs(frame_path)
        decompose_video(frame_path, video_input, skip)
    print("Searching...")
    image_inputs = [os.path.join(frame_path, filename)
        for filename in os.listdir(frame_path)]
    vid2tex, tex2vid = {}, {}
    run_clip(vid2tex, tex2vid, image_inputs, text_input, top_k)

def test(num_examples=5, top_k=1, skip=60):
    if not os.path.isfile(vid2tex_filename) or not os.path.isfile(tex2vid_filename):
        dataset_text_parser.export_descriptions()
    with open(vid2tex_filename) as f1, open(tex2vid_filename) as f2:
        vid2tex = json.load(f1)
        tex2vid = json.load(f2)
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
            images_list = dataset_processor.decompose_video(frame_path, video_input, skip)
        for image in images_list:
            image_inputs.append(image)
    image_features = get_image_embeddings(image_inputs)
    total_num_correct = 0
    with Bar('Testing examples...') as bar:
        for i in range(num_examples): #TODO: VECTORIZE THIS
            text_input = list(vid2tex.values())[i][0]
            text_features = get_text_embedding(text_input)
            total_num_correct += run_clip(
                vid2tex, tex2vid, image_inputs, text_input,
                image_features, text_features, top_k
            )
            bar.next()
    accuracy  = total_num_correct / num_examples
    print(f"\nOverall Accuracy: {accuracy}")

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
                image_features, text_features, top_k=5):
    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(top_k)
    # Print the result
    print(f"\nUser input: {text_input}:")
    num_correct = 0
    for value, index in zip(values, indices):
        video_name = image_inputs[index].split("/")[1]
        video_name = video_name[:video_name.index(".")]
        num_correct += int(video_name == tex2vid[text_input][0])
        print(f"Model prediction: video {video_name}. Depicts: {vid2tex[video_name][0]}")
        print(f"Frame: {image_inputs[index]:>16s}")
        print(f"Probability: {100 * value.item():.2f}%")
    return num_correct

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
    test(int(args.num_examples), int(args.top_k), int(args.skip))
    '''
    video_input = input("Enter filename of video you'd like to search: ")
    text_input = input(
        "Enter brief (few word) description of object you'd like to find: "
    )
    main(video_input, text_input, int(args.top_k), int(args.skip))
    '''
