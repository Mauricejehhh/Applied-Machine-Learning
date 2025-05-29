import torch.nn as nn
import torchvision.models as models


class BboxRegression(nn.Module):
    def __init__(self):
        super(BboxRegression, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 4)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.bbox_head(x)
        return x
