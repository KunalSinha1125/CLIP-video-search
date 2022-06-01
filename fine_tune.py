import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from data import Dataset
from tqdm import tqdm #pip install tqdm
from PIL import Image, ImageSequence
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def train(num_examples, batch_size, freeze, lr, num_epochs, keep, save_fps,
          folder="models/", filename="finetuned.pt"):

    #Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    if device == "cpu":
        model.float()

    #Freeze the necessary parameters
    num_freeze = int(len(list(model.parameters())) * freeze)
    for param in list(model.parameters())[:num_freeze]:
        param.requires_grad = False

    #Load the dataset
    train_dataset = Dataset(
        num_examples=num_examples, save_fps=save_fps, keep=keep, data_type="finetuned"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    #Define the loss and the optimization method
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2
    ) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    #Start training
    start = time.time()
    for epoch in tqdm(range(num_epochs), desc="Training"):
        for i, (images, texts, image_names, text_names) in enumerate(train_dataloader):
            #Zero out the gradients
            optimizer.zero_grad()

             #Get model predictions
            image_logits, text_logits = model(images, texts)

            #Get groundtruth
            text_names = np.array(text_names).reshape(1, len(text_names))
            image_groundtruth = torch.tensor((text_names == text_names.T) * 100, dtype=torch.float32)
            text_groundtruth = torch.transpose(image_groundtruth, 0, 1)

            #Compute loss
            total_loss = (loss(image_logits, image_groundtruth) + loss(text_logits, text_groundtruth)) / 2
            total_loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}")
            print(f"Loss: {total_loss}\n")

    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    model_dir = os.path.join(folder, filename)
    print(f"Saving model as {model_dir}")
    torch.save(model.state_dict(), os.path.join(folder, filename))


def load(folder="models", filename="finetuned.pt", device="cpu"):
    with open(os.path.join(folder, filename), 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location=device)
        model = clip.model.build_model(state_dict).to(device)
        if str(device) == "cpu":
            model.float()
        return model

#Print out model architecture
def describe_architecture(model):

    #Iterate through each child and print its contents
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is -")
        print(child)
        child_counter += 1

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_examples',
                        default='20',
                        help='How many examples to train on?')
    parser.add_argument('--batch_size',
                        default='64',
                        help='How many examples per batch?')
    parser.add_argument('--freeze',
                        default='.9',
                        help='What proportion of weights should be frozen?')
    parser.add_argument('--lr',
                        default='5e-5',
                        help='What is the learning rate?')
    parser.add_argument('--num_epochs',
                        default='30',
                        help='How many epochs?')
    parser.add_argument('--keep',
                        action='store_true',
                        help='Indicates that you dont want to resave the data')
    parser.add_argument('--save_fps',
                        default=1,
                        help='How many frames per second to save')
    args = parser.parse_args()
    train(int(args.num_examples), int(args.batch_size), float(args.freeze),
          float(args.lr), int(args.num_epochs), args.keep, int(args.save_fps))
