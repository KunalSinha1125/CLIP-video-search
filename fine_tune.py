import clip
import os
import torch
import torch.nn as nn

model_type = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"

#Fine-tune the model
def train_finetuned(model_type, device):

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
            old_layer.in_features + 100,
            old_layer.out_features,
            bias=True
        )
        module.resblocks[11].mlp.c_proj = new_layer
    #Note that requires_grad=True for new blocks by default!

    #Verify the arhitecture was altered successfully
    describe_architecture(model)


#Print out model architecture
def describe_architecture(model):

    #Iterate through each child and print its contents
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is -")
        print(child)
        child_counter += 1

if __name__ == "__main__":
    train_finetuned()
