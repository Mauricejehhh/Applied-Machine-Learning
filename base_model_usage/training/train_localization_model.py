"""
Dataset folder should be located under
Applied-Machine-Learning/project_name/data/tt100k_2021.
Make sure this is the case, otherwise it will not work.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from project_name.models.localization_base_model import CNNDetector
from project_name.data.dataset_loader import TT100KDataset
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


root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + 'train/ids.txt'
model_path = os.getcwd() + '/models/localization_model.pth'

if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    filtered_imgs = {img_id: img_data
                     for img_id, img_data in annos['imgs'].items()
                     if img_id in ids}

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# This dataset must return (image_tensor, bounding_box_tensor)
tt100k_data = TT100KDataset(filtered_annotations, root, transforms)

t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size

train_split, val_split = random_split(tt100k_data, [t_size, v_size])
t_loader = DataLoader(train_split, 32, shuffle=True)
v_loader = DataLoader(val_split, 32, shuffle=True)

epochs = 5
lr = 0.001

print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CNNDetector()  # assumes output is 4 values for the bounding box
model = model.to(device)
loss_fn = nn.MSELoss()  # regression loss for bounding boxes
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    model.train()
    running_tloss = 0.0
    running_vloss = 0.0

    for i, data in enumerate(tqdm(t_loader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        t_loss = loss_fn(outputs, targets)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'\n batch {i}: last loss: {running_tloss / 10:.4f}')
            running_tloss = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(v_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_vloss += loss.item()

        avg_loss = running_vloss / len(v_loader)
        print(f'Average validation loss: {avg_loss:.4f}')

torch.save(model.state_dict(), model_path)
print(f'Saved model to: {model_path}')
