import torch
import torch.nn as nn


class CNNDetector(nn.Module):
    """
    A simple CNN that predicts bounding boxes for signs in images.
    Output: [batch_size, 4] -> (xmin, ymin, xmax, ymax) for each image.
    """
    def __init__(self):
        super(CNNDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 values for the bounding box
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
