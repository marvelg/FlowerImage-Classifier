#Imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


def load_data(location):
    #Define Transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    #Define where to find data directory
    data_dir = location
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Set dataset and apply transforms
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)

    valid_dataset = datasets.ImageFolder(valid_dir, transform = test_valid_transforms)

    test_dataset = datasets.ImageFolder(test_dir, transform = test_valid_transforms)
    
    #Creates batches of data as data loader to load into NN
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)

    return train_dataset, trainloader, validloader, testloader
   
    
    
def neural_network(get_structure = 'vgg16', hidden_units = 4096, dropout = 0.3):
    
    #Choose model
    try:
        model = getattr(models, get_structure)(pretrained = True)
    except AttributeError:
        return "Invalid Model, choose a different model"
        
    #Prevents backprop through the entire model    
    for param in model.parameters():
        param.requires_grad = False
        
    #Creates classifier section of model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units, bias = True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p = dropout)),
        ('fc2', nn.Linear(hidden_units, 102, bias = True)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier
    
    return model
   
    
      


def train_model(model, trainloader, validloader,
                epochs = 3, learnrate = 0.001, devices = "gpu"):
    #Choose device
    if devices.lower() == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif devices.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        return print('That is not a valid device, pick gpu or cpu')

    #Training the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learnrate)

    model.to(device)

    steps = 0
    print_every = 20

    for e in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                accuracy = 0
                vlost = 0
                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        output = model.forward(inputs)
                        vlost += criterion(output, labels)
                        
                        probabilities = torch.exp(output)
                        top_p, top_class = probabilities.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
            
                        accuracy += torch.mean(equals.type(torch.FloatTensor))
                
                    vlost = vlost/len(validloader)
            
                print("Accuracy: {}%".format(accuracy * 100/len(validloader))
                      ,"Validation Lost: {}".format(vlost)
                      ,"Training Loss: {}".format(running_loss/print_every))
                
                running_loss = 0
                model.train()

#train_model(model, trainloader, validloader, epochs = 3, learnrate = 0.001, devices = "gpu"):
                
                
def save_checkpoint(path, structure, train_dataset, model, 
                    hidden_units = 4096, dropout = 0.3):
    
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'input_size': 25088,
                  'hidden_units': hidden_units,
                  'output_size': 102,
                  'dropout': dropout,
                  'structure' : structure,
                  'state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx}

    torch.save(checkpoint, path)
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = getattr(models, checkpoint['structure'])(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(checkpoint['input_size'],checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p = checkpoint['dropout'])),
        ('fc2', nn.Linear(checkpoint['hidden_units'], checkpoint['output_size'])),
        ('output', nn.LogSoftmax(dim = 1))
        ]))
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    #Resized images to 256 being the shortest side
    resized_side = 256
    width, height = im.size
    if im.size[0] > im.size[1]:
        width_resize = resized_side
        scale_percent = width_resize/width
        height_resize = int(height * scale_percent)
        im = im.resize((width_resize, height_resize))
    else:
        height_resize = resized_side
        scale_percent = height_resize/height
        width_resize = int(width * scale_percent)
        im = im.resize((width_resize, height_resize))
    
    #Center Crop Image

    im = transforms.CenterCrop(224)(im)
    
    #converts to numpy array and transforms color from 255 to 0 - 1
    im = np.array(im)/255
    
    #Normalizes image and convert to tensor
    Normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    im = Normalize(im)

    return im


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, devices = "gpu", topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if devices.lower() == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif devices.lower() == 'cpu':
        device = torch.device('cpu')
    else:
        return print('That is not a valid device, pick gpu or cpu')
    
    img = process_image(image_path).unsqueeze_(0).float()
    img = img.to(device)
    model.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model.forward(img)
        
        probabilities = torch.exp(output)
        
        top_p, top_class = probabilities.topk(topk , dim = 1)
        
    return top_p.tolist(), top_class.tolist()




def sanity_check(path, model, devices = "gpu", topk = 5):
    #inverse dictionary of train_dataset.class_to_idx (Assigns folder name(index) to ordered values (value))
    dictfolder = model.class_to_idx
    inv_dict = {value: index for index, value in dictfolder.items()}
    
    prediction = predict(path, model, devices, topk)
    image = process_image(path)
    
    ax = imshow(image)
    ax.axis('off')
    ax.set_title(cat_to_name[inv_dict[(prediction[1][0][0])]])
    
    a = np.array(prediction[0][0])
    b = [cat_to_name[inv_dict[index]] for index in prediction[1][0]]
    
    fig, ax1 = plt.subplots(figsize= (5,5))
    
    ax1.barh(b, a)