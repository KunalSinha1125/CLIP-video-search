import os
import torch
import clip
from data import Dataset, FPS
import random
from sklearn.cluster import AgglomerativeClustering

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clusterKeyFrames(dataset, batch_size, save_fps=1):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #get image features
    image_features_list = []
    for images, texts in dataloader:
        image_features_list += model.encode_image(images).tolist()

    num_frames_to_save = int(dataset.total_num_frames * (save_fps / FPS))
    hc = AgglomerativeClustering(
        n_clusters=num_frames_to_save,
        affinity='euclidean', linkage='ward', distance_threshold=None
    )

    y_hc = hc.fit_predict(image_features_list)
    assignments = set(y_hc) # unique assignments
    assignDict = {}
    for assignment in assignments:
        if assignment not in assignDict.keys():
            assignDict[assignment] = []
        for i in range(len(y_hc)):
            if y_hc[i] == assignment:
                assignDict[assignment].append(i)
    keyFrames = []
    for assign in assignDict.keys():
        keyFrames += random.sample(assignDict[assign], 1)
    return keyFrames
