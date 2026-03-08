import torch
import torch.nn as nn
from torchvision import models

# VGG的标准化参数 (ImageNet)
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
VGG_STD = torch.tensor([0.229, 0.224, 0.225])

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], weights=None):
        """
        Perceptual Loss.
        注意：这个类不接受 depth_range。
        它假设输入的深度图会被归一化到 [0, 1] 范围。
        """
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练的VGG16，并设置为评估模式
        vgg = models.vgg16(pretrained=True).features.eval()

        # 禁用所有 ReLU 的 inplace，防止破坏计算图
        for m in vgg.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        
        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            # 确保 l 是整数
            block = nn.Sequential(*[vgg[i] for i in range(prev, int(l))])
            self.blocks.append(block)
            prev = int(l)

        # 冻结VGG的所有参数
        for param in self.parameters():
            param.requires_grad = False
            
        self.weights = weights if weights is not None else [1.0] * len(self.blocks)
        
        # 将VGG的标准化参数注册为buffer，以便自动移动到GPU
        self.register_buffer('mean', VGG_MEAN.view(1, 3, 1, 1))
        self.register_buffer('std', VGG_STD.view(1, 3, 1, 1))

    def _preprocess(self, x):
        """
        将单通道的深度图（已归一化到[0,1]）转换为VGG期望的输入。
        """
        # 1. 复制成3通道
        x = x.repeat(1, 3, 1, 1)
        # 2. 使用ImageNet的均值和方差进行标准化
        return (x - self.mean) / self.std

    def forward(self, predict, gt):
        """
        计算感知损失。
        重要：输入的 predict 和 gt 应该是原始的、具有物理尺度的深度图！
        归一化步骤应在此函数内部完成。
        """
        
        # --- 关键：稳定且有意义的归一化 ---
        # 假设我们关注的深度范围是 [0, 10]。我们将这个范围线性映射到 [0, 1]。
        # 这种归一化是固定的，不会因batch而变。
        # 注意：这里我们简单地除以10.0，如果您有更精确的范围，可以使用 min-max 归一化。
        normed_pred = torch.clamp(predict / 10.0, 0.0, 1.0)
        normed_gt = torch.clamp(gt / 10.0, 0.0, 1.0)
        # ------------------------------------

        pred_vgg_input = self._preprocess(normed_pred)
        gt_vgg_input = self._preprocess(normed_gt)

        loss = 0.0
        
        # 传递通过VGG的各个block
        for i, block in enumerate(self.blocks):
            pred_vgg_input = block(pred_vgg_input)
            gt_vgg_input = block(gt_vgg_input)
            loss += self.weights[i] * torch.nn.functional.l1_loss(pred_vgg_input, gt_vgg_input)
            
        return loss
