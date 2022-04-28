import os
import torch
import clip #pip install clip
from PIL import Image, ImageSequence
import cv2 #pip install opencv-python

frame_dir = 'frames/'
video_dir = 'videos/'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main(video_input):
    image_inputs = decompose_video(video_input)
    text_input = 'fish'
    run_clip(image_inputs, text_input)

def decompose_video(video_input, filetype='png'):
    video_path = os.path.join(video_dir, video_input)
    capture = cv2.VideoCapture(video_path)
    frame_path = os.path.join(frame_dir, video_input)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    frame_number = 0
    frame_list = []
    while True:
        success, frame = capture.read()
        if success:
            filename = os.path.join(frame_path, str(frame_number)+'.'+filetype)
            cv2.imwrite(filename, frame)
            frame_list.append(filename)
        else:
            break
        frame_number += 1
    capture.release()
    return frame_list

def run_clip(image_inputs, text_input, top_k=2):

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
    video_input = "beach_sunset.mp4"
    main(video_input)
