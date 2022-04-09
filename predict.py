import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import supplement_functions as funct


parser = argparse.ArgumentParser()
parser.add_argument('img', type = str)
parser.add_argument('checkpoint', type= str)
parser.add_argument('--top_k', type = int, default = 3)
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json')
parser.add_argument('--gpu', default = "gpu")

args = parser.parse_args()
image_path = args.img
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
device = args.gpu

model = funct.load_checkpoint(checkpoint)

with open(category_names, 'r') as json_file:
    id_to_name = json.load(json_file)

dictfolder = model.class_to_idx
inv_dict = {value: index for index, value in dictfolder.items()}

prediction = funct.predict(image_path, model, device, top_k) 
probabilities = np.array(prediction[0][0]) * 100
labels = [id_to_name[inv_dict[index]] for index in prediction[1][0]]

print(prediction)

i=0
while i < top_k:
    print("{} with a probability of {}%".format(labels[i], probabilities[i]))
    i += 1

