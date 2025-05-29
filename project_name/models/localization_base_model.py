import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BboxRegression(nn.Module):
    def __init__(self, num_boxes):
        super(BboxRegression, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.num_boxes = num_boxes
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_boxes * 4)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.bbox_head(x)
        x = x.view(x.size(0), -1, 4)
        return x
