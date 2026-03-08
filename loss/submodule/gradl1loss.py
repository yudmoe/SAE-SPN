import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class gradL1Loss(nn.Module):
    def __init__(self, depth_range=None):
        super(gradL1Loss, self).__init__()
        
        # 确保 depth_range 是有效的
        if depth_range is None or len(depth_range) != 2:
            raise ValueError("depth_range 必须是长度为 2 的元组")
        
        self.depth_range = depth_range

    def forward(self, predict, gt):
        assert predict.size() == gt.size(), f"Size mismatch: predict.size()={predict.size()}, gt.size()={gt.size()}"
        B,C,H,W = predict.size()
         # 获取 device，避免默认 CPU 分配
        device = predict.device
        # 计算梯度（横向差分和纵向差分）
        grad_pred_x = predict[:, :, :, 1:] - predict[:, :, :, :-1]
        grad_pred_y = predict[:, :, 1:, :] - predict[:, :, :-1, :]
        grad_gt_x = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        grad_gt_y = gt[:, :, 1:, :] - gt[:, :, :-1, :]

        mask = (gt > 1e-3).type_as(gt).detach()
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        diff_x = torch.abs(grad_pred_x - grad_gt_x) * mask_x
        diff_y = torch.abs(grad_pred_y - grad_gt_y) * mask_y

        num_valid_x = mask_x.sum(dim=[1, 2, 3])
        num_valid_y = mask_y.sum(dim=[1, 2, 3])

        loss_x = diff_x.sum(dim=[1, 2, 3]) / (num_valid_x + 1e-8)
        loss_y = diff_y.sum(dim=[1, 2, 3]) / (num_valid_y + 1e-8)

        # save_path = "/data2/zzy/lightning_experiments/version_1852_V7306_addGradloss_210/checkpoints/visgrad"
        # visulize_Grad(grad_pred_x ,os.path.join(save_path, str("grad_pred_x.png")),  min_=grad_gt_x.min() , max_=grad_gt_x.max() )
        # visulize_Grad(grad_pred_y ,os.path.join(save_path, str("grad_pred_y.png")),  min_=grad_gt_y.min() , max_=grad_gt_y.max() )
        # visulize_Grad(grad_gt_x ,os.path.join(save_path, str("grad_gt_x.png")),  min_=grad_gt_x.min() , max_=grad_gt_x.max() )
        # visulize_Grad(grad_gt_y ,os.path.join(save_path, str("grad_gt_y.png")),  min_=grad_gt_y.min() , max_=grad_gt_y.max() )
        # visulize_Grad(diff_x ,os.path.join(save_path, str("diff_x.png")))
        # visulize_Grad(diff_y ,os.path.join(save_path, str("diff_y.png")))
        # visulize_Grad(predict ,os.path.join(save_path, str("predict.png")))
        # visulize_Grad(gt ,os.path.join(save_path, str("gt.png")))


        return (loss_x + loss_y).mean()

        # grad_pred = torch.zeros((B,2,H,W)).to(predict.device)
        # grad_pred[:, 0, :, 1:] = predict[:, 0, :, 1:] - predict[:, 0, :, :-1]
        # grad_pred[:, 1, 1:, :] = predict[:, 0, 1:, :] - predict[:, 0, :-1, :]

        # grad_gt = torch.zeros((B,2,H,W)).to(predict.device)
        # grad_gt[:, 0, :, 1:] = gt[:, 0, :, 1:] - gt[:, 0, :, :-1]
        # grad_gt[:, 1, 1:, :] = gt[:, 0, 1:, :] - gt[:, 0, :-1, :]
        # mask = (gt > 1e-3).type_as(gt).detach()

        # grad_mask = torch.zeros((B,2,H,W)).to(predict.device)
        # grad_mask[:, 0, :, 1:] = mask[:, 0, :, 1:] * mask[:, 0, :, :-1]
        # grad_mask[:, 1, 1:, :] = mask[:, 0, 1:, :] * mask[:, 0, :-1, :]
        

        # num_valid = torch.sum(grad_mask, dim=[1, 2, 3])

        # i_loss = torch.abs(grad_pred - grad_gt) * grad_mask
        # i_loss = torch.sum(i_loss, dim=[1, 2, 3]) / (num_valid + 1e-8)

        # return i_loss.mean()


def visulize_Grad(grad, save_path = "/data2/zzy/lightning_experiments/version_1852_V7306_addGradloss_210/checkpoints/visgrad",  min_ =None, max_ =None):

    def norm_save_(dep, path , min_=-10, max_=10):
        cm_tmp = plt.get_cmap('jet')
        if min_==None:
            min_ = dep.min()
        if max_==None:
            max_ = dep.max()
        norm_ = plt.Normalize(vmin=min_, vmax=max_)
        temp = (cm_tmp(norm_(dep)))
        plt.imsave(path, temp)

    grad = grad[0,0].data.cpu().numpy() 

    norm_save_(grad, save_path, min_ , max_ )