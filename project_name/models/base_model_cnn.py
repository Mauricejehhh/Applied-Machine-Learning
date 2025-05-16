import torch.nn as nn


# Design choises motivated by torch tutorial:
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNNClassifier(nn.Module):
    def __init__(self, number_of_classes):
        super.__init__()
        self.convolution_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, x):
        x = self.convolution_layer(x)
        x = self.classifier(x)
        return x