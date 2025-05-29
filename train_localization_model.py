import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.patches as patches
import albumentations as A
from project_name.models.localization_base_model import BboxRegression
from project_name.data.dataset_loader import TT100KDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


# These work for me, I am unsure whether they could work for you.
# To be sure, run this file from the /Applied-Machine-Learning
# directory, it should theoretically work.
root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + 'train/ids.txt'
model_path = os.getcwd() + '/models/locali_model.pth'

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
    print(f'Found {len(filtered_imgs)} training images.')

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

# Set up transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# Load dataset
tt100k_data = TT100KDataset(filtered_annotations, root, transform)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size
train_split, val_split = random_split(tt100k_data, [t_size, v_size])
subset = torch.utils.data.Subset(train_split, range(4))
loader = DataLoader(subset, batch_size=4, collate_fn=collate_fn)
t_loader = DataLoader(train_split,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=collate_fn)

v_loader = DataLoader(val_split,
                      batch_size=4,
                      shuffle=False,
                      collate_fn=collate_fn)

# Initialize training loop parameters
epochs = 20
lr = 0.001

# Set device (cuda or cpu) and initialize bbox regressor
print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BboxRegression()
model = model.to(device)

# Define loss function and optimizer
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    model.train()
    running_tloss = 0.0

    for i, (images, targets) in enumerate(tqdm(t_loader)):
        images = [img for img in images]
        labels = [t['boxes'] for t in targets]
        images = torch.stack(images).to(device)
        labels = torch.stack(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images)

        print(f"Output: {outputs[0].detach().cpu().numpy()}")
        print(f"Label : {labels[0].detach().cpu().numpy()}")
        t_loss = loss_fn(outputs, labels)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'\nBatch {i}: Avg Loss = {running_tloss / 10:.6f}')
            running_tloss = 0.0

    # Validation loop
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for images, targets in tqdm(v_loader):
    #         images = [img for img in images]
    #         labels = [t['boxes'] for t in targets]
    #         images = torch.stack(images).to(device)
    #         labels = torch.stack(labels).to(device)
    #         outputs = model(images)
    #         val_loss = loss_fn(outputs, labels)
    #         running_tloss += val_loss.item()

    # print(f'Validation Loss: {val_loss / len(v_loader):.4f}')

# torch.save(model.state_dict(), model_path)
# print(f'Saved model to: {model_path}')

# model = BboxRegression()
# model = model.to(device)
# m_state_dict = torch.load(model_path, weights_only=True)
# model.load_state_dict(m_state_dict)

# Pick one sample from the validation set
model.eval()
sample_image, sample_target = train_split.__getitem__(4)
image_tensor = sample_image.to(device).unsqueeze(0)
print(sample_image.shape)
with torch.no_grad():
    prediction = model(image_tensor).squeeze(0)

print(prediction.shape)
# Convert image to NumPy
denorm_image = inv_normalize(image_tensor)
denorm_image = denorm_image.squeeze(0)
npimg = denorm_image.numpy()
npimg = np.transpose(denorm_image, (1, 2, 0))
print(npimg.shape)

# Plot
fig, ax = plt.subplots(1)
ax.imshow(npimg)

x1, y1, x2, y2 = prediction.cpu() * torch.tensor([224, 224, 224, 224])
w, h = x2 - x1, y2 - y1
rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
ax.add_patch(rect)

x1, y1, x2, y2 = sample_target['boxes'] * torch.tensor([224, 224, 224, 224])
w, h = x2 - x1, y2 - y1
rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none')
ax.add_patch(rect)

plt.title("Red: Prediction | Green: Ground Truth")
plt.show()
