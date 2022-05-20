import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

#Fine-tune the model
def run_finetuning(model_type="ViT-B/32", device="cpu", loss_fn="cross_entropy",
                   lr=0.001, momentum=0.9, num_epochs=25, print_arch=False):

    #Load the model
    model, preprocess = clip.load(model_type, device=device)

    #Freeze the old layers
    for param in model.parameters():
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
    criterion = None
    if loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    trained_model = train_model(model, criterion, optimizer, num_epochs)

    return trained_model


def train_model(model, criterion, optimizer, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_finetuning(device=device)
