"""
Dataset folder should be located under
Applied-Machine-Learning/project_name/data/tt100k_2021.
Make sure this is the case, otherwise it will not work.
"""
import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from models.base_model_cnn import CNNClassifier
from features.dataset_loader import TT100KDataset, TT100KSignDataset
from torch.utils.data import random_split, DataLoader


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# These work for me, I am unsure whether they could work for you.
# To be sure, run this file from the /Applied-Machine-Learning
# directory, it should theoretically work.
root = os.getcwd() + '/project_name/data/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + '/train/ids.txt'
model_path = os.getcwd() + '/project_name/models/model.pth'

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
# tt100k_data = TT100KDataset(filtered_annotations, root)
transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

tt100k_data = TT100KSignDataset(filtered_annotations, root, transforms)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size

# Initialize data loaders for testing and validations sets
train_split, val_split = random_split(tt100k_data, [t_size, v_size])
# t_small, v_small = random_split(tt100k_data, [100, len(tt100k_data) - 100])
t_loader = DataLoader(train_split, 32, shuffle=True)
v_loader = DataLoader(val_split, 32, shuffle=True)

# Initialize training loop parameters
epochs = 1
lr = 0.001

# Initialize model, optimizer etc.
print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_of_classes = len(tt100k_data.annotations['types'])
model = CNNClassifier(num_of_classes)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    # Set model to training mode
    model.train()
    running_tloss = 0.0
    running_vloss = 0.0

    for i, data in enumerate(tqdm(t_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # img_grid = torchvision.utils.make_grid(inputs)
        # matplotlib_imshow(img_grid, one_channel=True)
        # plt.show()
        t_loss = loss_fn(outputs, labels)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'\n batch {i}: last loss: {running_tloss / 10}')
            running_tloss = 0

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(v_loader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_vloss += loss
        avg_loss = running_vloss / len(v_loader)
        print(f'Average validation loss: {avg_loss:.4f}')

torch.save(model.state_dict(), model_path)
print(f'Saved model to: {model_path}')

# # Loading a model:
# new_model = CNNClassifier(num_of_classes)
# m_state_dict = torch.load(model_path, weights_only=True)
# new_model = new_model.to(device)
# new_model.load_state_dict(m_state_dict)

# # Dumb testing thing:
# dataiter = iter(t_loader)
# images, labels = next(dataiter)
# img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=True)
# plt.show()
# outputs = new_model(images)
# _, predicted = torch.max(outputs, 1)
# print(f'Ground truth: {[tt100k_data.idx_to_label[label.item()] for label in labels]}')
# print(f'Predicted: {[tt100k_data.idx_to_label[pred.item()] for pred in predicted]}')