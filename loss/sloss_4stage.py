import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.submodule.l1l2loss import L1Loss,L2Loss

def down_sample(depth,stride):
    C = (depth > 0).float()
    output = F.avg_pool2d(depth, stride, stride) / (torch.add(F.avg_pool2d(C, stride, stride), 1e-5))
    return  output


import settings_NYU as settings
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
        y_inter = output["list_feat"]

        l1 = self.l1loss(pred, gt)
        l2 = self.l2loss(pred, gt)
                
        loss_weight = None
        if epoch<=settings.downLR1:
            loss_weight = 0.5
        elif epoch<=settings.downLR2:
            loss_weight = 0.2

        if loss_weight:
            new_gt = gt
            try:
                prop_time = len(y_inter)
                half_pred= y_inter[int(prop_time*3/4)-1]
                quarter_pred = y_inter[int(prop_time*2/4)-1]
                engihth_pred  = y_inter[int(prop_time/4)-1]
            except:
                half_pred = y_inter[2]
                quarter_pred = y_inter[1]
                engihth_pred = y_inter[0]
                
            half_gt = down_sample(new_gt,2)
            quarter_gt = down_sample(new_gt,4)
            # engihth_gt = down_sample(new_gt,8)
            C = (gt > 0).float()
            engihth_gt = F.avg_pool2d(new_gt, kernel_size = 8, stride = 8, padding=(2,0)) / (F.avg_pool2d(C, kernel_size = 8, stride = 8, padding=(2,0)) + 0.0001)


            l1 += self.l1loss(half_pred, half_gt) * loss_weight
            l1 += self.l1loss(quarter_pred, quarter_gt) * loss_weight
            l1 += self.l1loss(engihth_pred, engihth_gt) * loss_weight
            l2 += self.l2loss(half_pred, half_gt) * loss_weight
            l2 += self.l2loss(quarter_pred, quarter_gt) * loss_weight
            l2 += self.l2loss(engihth_pred, engihth_gt) * loss_weight
        


        return self.w1 * l1 + self.w2 * l2
    

