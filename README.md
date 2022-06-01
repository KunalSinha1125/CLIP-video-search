# CLIP-video-search
We are using CLIP for Ad Hoc Video Search! Enter in the name of some object, and the model will return a timestamp of the scene where the object appears.

Guide to the files:
* test.py: performs model testing
    * Use command line flags to specify model type, frame extraction method, number of examples, batch size, etc.
* fine_tune.py: fine-tunes the model
    * Use command lines flags to control hparams such as number of examples, number of epochs, number of frames to freeze, learning rate, etc.
* data.py: defines the Dataset class passed into the DataLoader
    * Dataset class has fields representing each image + text in the dataset along with its filename
    * File has methods for returning elements from Dataset, deleting elements, and breaking down a video into frames
* h_clustering.py: performs hierarchical clustering to extract keyframes

Sources consulted during coding:

test.py: https://github.com/openai/CLIP

decompose_video(): https://techtutorialsx.com/2021/04/29/python-opencv-splitting-video-frames/

fine_tune.py: https://github.com/openai/CLIP/issues/83

h_clustering.py: https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318

data.py: https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
