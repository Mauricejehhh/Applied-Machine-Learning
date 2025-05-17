"""
Dataset folder should be located under
Applied-Machine-Learning/project_name/data/tt100k_2021.
Make sure this is the case, otherwise it will not work.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from models.base_model_cnn import CNNClassifier
from features.dataset_loader import TT100KDataset
from torch.utils.data import random_split, DataLoader


# These work for me, I am unsure whether they could work for you.
# To be sure, run this file from the /Applied-Machine-Learning
# directory, it should theoretically work.
root = os.getcwd() + '/project_name/data/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + '/train/ids.txt'

# Find all training ids from the ids.txt file in train/
# This is bound to change as custom splits will be needed.
if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    # Filter annotations file for training ids only
    filtered_imgs = {img_id: img_data
                     for img_id, img_data in annos['imgs'].items()
                     if img_id in ids}

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

# Train split and validation split should be decided later,
# these are just values for now
tt100k_data = TT100KDataset(filtered_annotations, root)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size

# Initialize data loaders for testing and validations sets
train_split, val_split = random_split(tt100k_data, [t_size, v_size])
t_loader = DataLoader(tt100k_data, 4, shuffle=True)
v_loader = DataLoader(tt100k_data, 4, shuffle=True)

# Initialize model, optimizer etc.
num_of_classes = len(tt100k_data.annotations['types'])
model = CNNClassifier(num_of_classes)


# Temporary function for plotting
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    if one_channel:
        plt.imshow(img, cmap="Greys")
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


dataiter = iter(t_loader)
images = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
