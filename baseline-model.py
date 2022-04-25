import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main():
    image_name = 'cs permutations.png'
    text_names = ['a diagram', 'a dog', 'a cat']
    probs = run_clip(image_name, text_names)
    print(probs)

def run_clip(image_name, text_names):

    image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)
    text = clip.tokenize(text_names).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

if __name__ == "__main__":
    main()
