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

def train(folder="models/", filename="finetuned.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    if device == "cpu":
        model.float()

    freeze = 0.95
    num_freeze = int(len(list(model.parameters())) * freeze)
    for param in list(model.parameters())[:num_freeze]:
        param.requires_grad = False

    num_examples = 1
    batch_size = 64
    shuffle = True
    train_dataset = Dataset(num_examples)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=5e-5, betas=(0.9,0.98),
        eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    num_epochs = 5

    start = time.time()
    for epoch in tqdm(range(num_epochs)):
        for images, texts in train_dataloader :
            optimizer.zero_grad()

            #list_image,list_txt = batch #list_images is list of image in numpy array(np.uint8), or list of PIL images

            #images= torch.stack([preprocess(Image.open(img)) for img in list_image],dim=0).to(device) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            #texts = clip.tokenize(list_txt).to(device)

            logits_per_image, logits_per_text = model(images, texts)

            batch_size = images.shape[0]
            ground_truth = torch.arange(batch_size,dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            print(f"Epoch: {epoch}")
            print(f"Loss: {total_loss}\n")
            total_loss.backward()
            if device == "cpu":
                optimizer.step()

    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
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
    train()#run_finetuning()

'''
device = "cuda" if torch.cuda.is_available() else "cpu"

#Fine-tune the model
def run_finetuning(model="ViT-B/32", folder='models', filename="finetuned.pt", loss_fn="mse_loss",
                   lr=0.01, momentum=0.9, num_epochs=30, print_arch=False,
                   batch_size=64, shuffle=True, freeze=.75):

    #Load the model
    model, preprocess = clip.load(model, device=device)

    #Freeze the old layers
    num_freeze = int(len(list(model.parameters())) * freeze)
    for param in list(model.parameters())[:num_freeze]:
        param.requires_grad = False

    #For the visual and text transformer,
    #Replace the last FC layer inside the last residual block
    modules_list = [
        model._modules['visual'].transformer, model._modules['transformer']
    ]
    for module in modules_list:
        old_layer = module.resblocks[11].mlp.c_proj
        new_layer = nn.Linear(
            old_layer.in_features,
            old_layer.out_features,
            bias=True
        )
        module.resblocks[11].mlp.c_proj = new_layer
    #Note that requires_grad=True for new blocks by default!

    #Verify the arhitecture was altered successfully
    if print_arch:
        describe_architecture(model)

    #Run training
    dataloaders = load_dataset(batch_size, shuffle)
    criterion = None
    if loss_fn == "mse_loss":
        criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trained_model = train_model(
        model, folder, filename, dataloaders, criterion, optimizer, num_epochs
    )
    return trained_model

def load_dataset(batch_size, shuffle):
    train_dataset = Dataset()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return {"train": train_loader}

def train_model(model, folder, filename, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')#best_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Epoch:"):
        #print(f'Epoch {epoch}/{num_epochs - 1}')
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0

            # Iterate over data.
            num_examples = 0
            for i, (images, texts) in enumerate(dataloaders[phase]):
                images = images.to(device)
                texts = texts.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, texts)[0]
                    _, preds = torch.max(outputs, 1)
                    pred_embeds = torch.column_stack([texts[p] for p in preds]).to(torch.float)
                    pred_embeds = torch.t(pred_embeds)
                    loss = criterion(pred_embeds, texts.to(torch.float))
                    #loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.requires_grad = True
                        loss.backward()
                        optimizer.step()

                # statistics
                num_examples += 1
                running_loss += loss.item() * images.size(0)
                #running_corrects += torch.sum(pred_embeds == texts.data)
            #if phase == 'train':
                #scheduler.step()

            epoch_loss = running_loss / num_examples#dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}') #Acc: {epoch_acc:.4f}')

            # deep copy the model
            if epoch_loss < best_loss:#phase == 'val' and epoch_acc > best_acc:
                best_loss = epoch_loss#best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f'Best val Acc: {best_acc:4f}')

    # load best model weights and save to file
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(folder, filename))
    return model
'''
