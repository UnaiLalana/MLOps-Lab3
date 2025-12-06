# model.py

from torchvision import models
import torch.nn as nn


def build_mobilenet_v2(num_classes: int):

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)

    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
