import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class BboxRegression(nn.Module):
    """
    A bounding box regression model using a frozen ResNet-50 backbone.

    The model predicts 4 normalized bounding box coordinates (xmin, ymin, xmax, ymax)
    from an input image. The output values are constrained between 0 and 1 using a Sigmoid.

    Backbone:
        - Pretrained ResNet-50 with final layers removed.
        - Global feature aggregation using AdaptiveAvgPool2d.
        - Fully connected head for 4-dimensional bounding box output.
    """
    def __init__(self) -> None:
        """
        Initialize the BboxRegression model with a pretrained ResNet-50 backbone.
        """
        super(BboxRegression, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Use all layers except the last two (avgpool and fc)
        self.base_model = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze the backbone to prevent training
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Bounding box prediction head
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 4),   # Output: [xmin, ymin, xmax, ymax]
            nn.Sigmoid()          # Normalize outputs to [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to predict bounding boxes.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, H, W)

        Returns:
            Tensor: Bounding box tensor of shape (batch_size, 4)
                    with normalized coordinates in [0, 1].
        """
        x = self.base_model(x)
        x = self.bbox_head(x)
        return x
