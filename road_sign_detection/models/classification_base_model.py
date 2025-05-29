import torch.nn as nn
from torch import Tensor


class CNNClassifier(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.

    Architecture:
        - 2 convolutional layers with ReLU and MaxPooling
        - 2 fully connected layers with ReLU
        - Final linear layer for classification

    Design inspired by PyTorch CIFAR-10 tutorial:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """
    def __init__(self, number_of_classes: int) -> None:
        """
        Initialize the CNN classifier.

        Args:
            number_of_classes (int): Number of output classes for classification.
        """
        super(CNNClassifier, self).__init__()

        self.convolution_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, 64, 64).

        Returns:
            Tensor: Output logits of shape (batch_size, number_of_classes).
        """
        x = self.convolution_layer(x)
        x = self.classifier(x)
        return x
