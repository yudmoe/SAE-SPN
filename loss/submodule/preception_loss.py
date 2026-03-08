import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torchvision import models


def preprocess_vgg(x):
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # 单通道变为 3 通道
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], weights=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features

        # 禁用所有 ReLU 的 inplace
        for m in vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False


        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev, int(l))])
            self.blocks.append(block)
            prev = l
        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights if weights else [1.0] * len(self.blocks)

    def forward(self, predict, gt):
        normed_pred = predict/predict.max()
        normed_gt = gt/gt.max()

        pred_features = preprocess_vgg(normed_pred)
        target_features = preprocess_vgg(normed_gt)

        loss = 0.0
        for i, block in enumerate(self.blocks):
            pred_features = block(pred_features)
            target_features = block(target_features)
            loss += self.weights[i] * torch.nn.functional.l1_loss(pred_features, target_features)
        return loss
