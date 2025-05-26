import torch.nn as nn
import torch.nn.functional as F
import torch


class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)   # -> (B,32,60,60)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)  # -> (B,64,26,26)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0) # -> (B,128,9,9)

        self.fc1 = nn.Linear(128 * 9 * 9, 2046)
        self.fc2 = nn.Linear(2046, 4)

    def forward(self, val):
        val = F.relu(self.conv1(val))       # [B, 32, 60, 60]
        val = F.max_pool2d(val, 2, 2)       # [B, 32, 30, 30]
        val = F.relu(self.conv2(val))       # [B, 64, 26, 26]
        val = F.max_pool2d(val, 2, 2)       # [B, 64, 13, 13]
        val = F.relu(self.conv3(val))       # [B, 128, 9, 9]

        val = val.view(val.size(0), -1)     # [B, 10368]
        val = F.dropout(F.relu(self.fc1(val)), p=0.5, training=self.training)
        val = torch.sigmoid(self.fc2(val))  # [B, 4] bounding boxes in [0,1] format
        return val

