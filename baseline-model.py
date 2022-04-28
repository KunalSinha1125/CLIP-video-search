import os
import torch
import clip #pip install clip
from PIL import Image, ImageSequence
import cv2 #pip install opencv-python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


frame_dir = 'frames/'
video_dir = 'videos/'
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    run_clip(image_inputs, text_input, top_k)

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

def run_clip(image_inputs, text_input, top_k=5):

    images = torch.cat( #Create a tensor representation for the images
        [preprocess(Image.open(img)).unsqueeze(0).to(device) for img in image_inputs]
    ).to(device)
    text = clip.tokenize(text_input).to(device) #Create a tensor representation for the text
    with torch.no_grad():
        image_features = model.encode_image(images) #Produce image embeddings
        text_features = model.encode_text(text) #Produce text embeddings

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(top_k)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{image_inputs[index]:>16s}: {100 * value.item():.2f}%")

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--top_k',
                        default=5,
                        help='How many frames to retrieve?')
    parser.add_argument('--skip',
                        default=15,
                        help='How many frames to skip while saving?')
    args = parser.parse_args()
    video_input = input("Enter filename of video you'd like to search: ")
    text_input = input(
        "Enter brief (few word) description of object you'd like to find: "
    )
    main(video_input, text_input, int(args.top_k), int(args.skip))
