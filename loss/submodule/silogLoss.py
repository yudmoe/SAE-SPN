import torch
import torch.nn as nn
import torch.nn.functional as F


class silogLoss(nn.Module):
    def __init__(self, depth_range=None, variance_focus=0.85):
        super(silogLoss, self).__init__()
        
        # 确保 depth_range 是有效的
        if depth_range is None or len(depth_range) != 2:
            raise ValueError("depth_range 必须是长度为 2 的元组")
        self.variance_focus = variance_focus
        self.depth_range = depth_range

    def forward(self, predict, gt):
        assert predict.size() == gt.size(), f"Size mismatch: predict.size()={predict.size()}, gt.size()={gt.size()}"

        
        # 根据 depth_range 创建 mask
        mask = (gt > self.depth_range[0]) & (gt < self.depth_range[1])

        # 计算对数深度
        log_prediction = torch.log(predict[mask])
        log_target = torch.log(gt[mask])

        # 计算对数差异
        log_diff = log_prediction - log_target
        
        # 计算SiLog Loss
        variance = torch.var(log_diff)
        mean = torch.mean(log_diff)
        
        silog_loss = torch.sqrt(variance + self.variance_focus * (mean**2))

        return silog_loss.mean()

