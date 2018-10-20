import matplotlib
matplotlib.use('tkagg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from loaders import ImageFolder
import mplcursors

image_dir = '../data/train_image'
label_file = '../dialogs/more_animals.json'
label_dict = None
label_names = None

with open(label_file) as json_data:
    data = json.load(json_data)
    label_dict = data['labeled_images']
    label_names = data['classes']
    json_data.close()

label_values = sorted(list(set(label_dict.values())))

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_dataset = ImageFolderWithPaths(image_dir, data_transform, filter_set=label_dict.keys())
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

dataset_size = len(image_dataset)
class_names = image_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes, test = next(iter(dataloader))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title='test')

model_final = torchvision.models.vgg19(pretrained=True)
for param in model_final.parameters():
    param.requires_grad = False
model_final = model_final.to(device)

model_final_conv = nn.Sequential(*list(model_final.children())[:-1])
model_final_conv = model_final_conv.to(device)

model_middle_conv = nn.Sequential(*list(list(model_final_conv.children())[0].children())[:28])
models = [model_final, model_final_conv, model_middle_conv]
model_names = ['Final Layer of VGG-19', 'Final Convolutional Layer of VGG-19', 'Intermediate Convolutional Layer of VGG-19']

models = models[:1]
model_names = model_names[:1]
for model, model_name in zip(models, model_names):
    all_outputs = []
    labels = []
    image_map = []
    iters = 0
    for inputs, _, image_file_name in image_dataset:
        file_ending = image_file_name.split('_')[-1]
        file_number = str(int(file_ending[:-4]))
        if file_number not in label_dict:
            continue
        inputs = inputs[None, :, :, :]
        inputs = inputs.to(device)
        outputs = model(inputs)
        np_output = outputs.detach().cpu().numpy()
        all_outputs.append(np_output)
        labels.append(label_dict[file_number])
        image_map.append(image_file_name)
        iters += 1
        if iters % 100 == 0:
            print(iters)
        if iters >= 1000:
            break

    print("Finished Convolution")
    print(all_outputs[0].shape)
    all_outputs = np.array(all_outputs)
    labels = np.array(labels)
    num_elems = all_outputs.shape[0]
    all_outputs = all_outputs.reshape((num_elems, -1))

    pca = PCA(n_components=50)
    pca_output = pca.fit_transform(all_outputs)
    # pca_output = all_outputs
    print("Finished PCA")
    X_embedded = TSNE(n_components=2).fit_transform(pca_output)
    print("Finished TSNE")

    plt.figure(figsize=(20, 20))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(X_embedded.shape[0]):
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], c=colors[labels[i]], label=image_map[i])
    # for i, label in zip(label_values, label_names):
    #     plt.scatter(X_embedded[labels == i,0], X_embedded[labels == i,1], label=label)
    plt.legend()
    plt.title(model_name)
    plt.savefig(model_name + '.png')
    mplcursors.cursor(hover=True)
    plt.show()
