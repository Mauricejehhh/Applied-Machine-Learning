"""
New evaluation file that loads a model trained through main.py
and uses an accuracy metric to evaluate the performance.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torchvision import transforms
from tqdm import tqdm
from project_name.models.classification_base_model import CNNClassifier
from project_name.data.dataset_loader import TT100KSignDataset
from torch.utils.data import DataLoader


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
root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_test_annotations.json'
ids_file = root + 'test/ids.txt'
model_path = os.getcwd() + '/models/model.pth'

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
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load testing data
tt100k_data = TT100KSignDataset(filtered_annotations, root, transforms)
test_loader = DataLoader(tt100k_data, 32, shuffle=True)

# Initialize model, optimizer etc.
print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_of_classes = len(tt100k_data.annotations['types'])

# Loading a model:
model = CNNClassifier(num_of_classes)
model = model.to(device)
m_state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(m_state_dict)

# Dumb testing thing:
dataiter = iter(test_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
plt.show()
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print(f'Ground truth: {[tt100k_data.idx_to_label[
    label.item()] for label in labels]}')
print(f'Predicted: {[tt100k_data.idx_to_label[
    pred.item()] for pred in predicted]}')

# Accuracy:
correct = 0
total = 0

with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(labels)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

print('Accuracy on testing data:', correct / total * 100)
