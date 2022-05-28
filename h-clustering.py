import os
import torch
import matplotlib.pyplot as plt
import clip
from data import Dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

num_examples = 5
batch_size = 32

#load the dataset
dataset = Dataset(num_examples = num_examples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

#get image features
image_features_list = []
for images, texts in dataloader:
    image_features_list += model.encode_image(images).tolist()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = None, affinity = 'euclidean', linkage = 'ward', distance_threshold=10)

y_hc = hc.fit_predict(image_features_list)
print(y_hc)

