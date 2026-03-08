import torch
import torch.nn as nn
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def __init__(self, depth_range):
        super(SSIMLoss, self).__init__()
        
        if depth_range is None or len(depth_range) != 2:
            raise ValueError("depth_range 必须是长度为 2 的元组或列表，例如 [0.1, 10.0]")
        
        # 我们关心的是数据的最大动态范围
        self.data_range = depth_range[1] - depth_range[0]
        # 如果您的数据保证最小值为0，也可以直接用 depth_range[1]
        # self.data_range = depth_range[1] 

    def forward(self, predict, gt):
        assert predict.size() == gt.size(), f"Size mismatch: predict.size()={predict.size()}, gt.size()={gt.size()}"
        
        # --- 不需要对 predict 和 gt 进行任何归一化 ---
        # 直接将原始的、具有物理尺度的深度图传入
        
        # 计算 SSIM Loss
        # 关键：data_range 必须与你的深度图的真实物理范围相匹配！
        # size_average=True 已被弃用，新版本建议使用 reduction='mean'，不过你当前的库可能还是旧版
        ssim_val = ssim(predict, gt, data_range=self.data_range, size_average=True)
        
        # SSIM 的范围是 [-1, 1]，但通常在 [0, 1] 之间。
        # 1 - ssim 是一个常用的、使其最小化的损失形式。
        return 1.0 - ssim_val

