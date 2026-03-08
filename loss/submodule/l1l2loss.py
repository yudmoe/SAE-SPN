import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(L1Loss, self).__init__()
        
        # 确保 depth_range 是有效的
        if depth_range is None or len(depth_range) != 2:
            raise ValueError("depth_range 必须是长度为 2 的元组")
        
        self.depth_range = depth_range

    def forward(self, predict, gt):
        assert predict.size() == gt.size(), f"Size mismatch: predict.size()={predict.size()}, gt.size()={gt.size()}"
        # 检查输入的大小是否一致
        # if predict.size() != gt.size():
        #     raise ValueError(f"预测和标签的尺寸不一致：预测尺寸 {predict.size()}，标签尺寸 {gt.size()}")

        # # 检查预测值 (predict) 是否包含 NaN
        # if torch.isnan(predict).any():
        #     print("预测值包含 NaN，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果有 NaN，返回 0 损失
        
        # # 检查标签 (gt) 是否包含 NaN 或 Inf
        # if torch.isnan(gt).any() or torch.isinf(gt).any():
        #     print("标签值包含 NaN 或 Inf，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果 gt 也有 NaN 或 Inf，返回 0 损失
        
        # 根据 depth_range 创建 mask
        mask = (gt > self.depth_range[0]) & (gt < self.depth_range[1])

        # # 如果 mask 为空，返回 0 损失
        # if mask.sum() == 0:
        #     print("没有符合条件的gt值，mask为空，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果 mask 为空，返回 0 损失

        # 计算 L1 损失，只计算符合条件的部分
        loss = torch.abs(predict[mask] - gt[mask])
        return loss.mean()

class L2Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(L2Loss, self).__init__()
        
        # 确保 depth_range 是有效的
        if depth_range is None or len(depth_range) != 2:
            raise ValueError("depth_range 必须是长度为 2 的元组")
        
        self.depth_range = depth_range

    def forward(self, predict, gt):
        assert predict.size() == gt.size(), f"Size mismatch: predict.size()={predict.size()}, gt.size()={gt.size()}"
        # 检查输入的大小是否一致
        # if predict.size() != gt.size():
        #     raise ValueError(f"预测和标签的尺寸不一致：预测尺寸 {predict.size()}，标签尺寸 {gt.size()}")

        # # 检查预测值 (predict) 是否包含 NaN
        # if torch.isnan(predict).any():
        #     print("预测值包含 NaN，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果有 NaN，返回 0 损失
        
        # # 检查标签 (gt) 是否包含 NaN 或 Inf
        # if torch.isnan(gt).any() or torch.isinf(gt).any():
        #     print("标签值包含 NaN 或 Inf，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果 gt 也有 NaN 或 Inf，返回 0 损失
        
        # 根据 depth_range 创建 mask
        mask = (gt > self.depth_range[0]) & (gt < self.depth_range[1])

        # # 如果 mask 为空，返回 0 损失
        # if mask.sum() == 0:
        #     print("没有符合条件的gt值，mask为空，跳过计算")
        #     return torch.zeros(1, device=predict.device, requires_grad=True)  # 如果 mask 为空，返回 0 损失

        # 计算 L2 损失，只计算符合条件的部分
        loss = torch.pow((predict[mask] - gt[mask]), 2)
        return loss.mean()

