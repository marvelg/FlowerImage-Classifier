# Imports here
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
parser.add_argument('data_dir')
parser.add_argument('--save_dir', default = './checkpoint.pth')
parser.add_argument("--arch", default = "vgg16")
parser.add_argument("--learning_rate", type = float, default = 0.001)
parser.add_argument("--hidden_units", type = int, default = 512)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--gpu', default="gpu")


args = parser.parse_args()
directory = args.data_dir
save_path = args.save_dir
learnrate = args.learning_rate
structure = args.arch
hidden_units = args.hidden_units
devices = args.gpu
epochs = args.epochs
dropout = args.dropout

train_dataset, trainloader, validloader, testloader = funct.load_data(directory)

model = funct.neural_network(structure, hidden_units, dropout)

funct.train_model(model, trainloader, validloader, epochs, learnrate, devices)

funct.save_checkpoint(save_path, structure, train_dataset, model, hidden_units, dropout)

funct.load_checkpoint(save_path)

print('Everything done - model trained and saved!')