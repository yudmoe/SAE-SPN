import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from loss.submodule.l1l2loss import L1Loss,L2Loss

def down_sample(depth,stride):
    C = (depth > 0).float()
    output = F.avg_pool2d(depth, stride, stride) / (torch.add(F.avg_pool2d(C, stride, stride), 1e-4))
    return  output


class SLoss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0, depth_range=None):
        super(SLoss, self).__init__()
        if depth_range is None:
            depth_range = [0.1, 100]
        self.l1loss = L1Loss(depth_range)
        self.l2loss = L2Loss(depth_range)
        self.w1 = w1
        self.w2 = w2

    def forward(self, output, gt, epoch):
        pred = output['pred']

        l1 = self.l1loss(pred, gt)
        l2 = self.l2loss(pred, gt)
                

        return self.w1 * l1 + self.w2 * l2

