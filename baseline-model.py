import os
import torch
import clip
from PIL import Image

image_dir = 'images/'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main():
    image_inputs = ['diagram.png', 'fish.png']
    text_input = 'fish'
    image_inputs = [os.path.join(image_dir, img) for img in image_inputs]
    run_clip(image_inputs, text_input)

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
    main()
