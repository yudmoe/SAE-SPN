import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
from PIL import Image
import cv2
import torch.nn.functional as F
from matplotlib.colors import ListedColormap, BoundaryNorm
cm_tmp = plt.get_cmap('Greys')
cm_plasma = plt.get_cmap('plasma')
cm = plt.get_cmap('jet')
cm_jet = plt.get_cmap('jet')



def add_2d_black_border(image_array, border_width=1):
    """
    给二维RGBA图像数组添加黑色边框
    
    参数:
        image_array: 形状为(h, w, 4)的RGBA数组
        border_width: 边框宽度（像素数）
    
    返回:
        添加了边框的新数组，形状为(h+2*border_width, w+2*border_width, 4)
    """
    h, w, c = image_array.shape
    # 确保是RGBA格式
    assert c == 4, "图像数组必须是RGBA格式（最后一维为4）"
    
    # 创建黑色像素（RGBA：0,0,0,1）
    black = np.array([0, 0, 0, 1.0], dtype=image_array.dtype)
    
    # 1. 添加上下边框（在高度方向扩展）
    # 生成形状为(border_width, w, 4)的边框
    top_border = np.tile(black, (border_width, w, 1))
    bottom_border = np.tile(black, (border_width, w, 1))
    # 拼接上下边框和原图
    with_tb = np.concatenate([top_border, image_array, bottom_border], axis=0)
    
    # 2. 添加左右边框（在宽度方向扩展）
    # 此时高度已变为h + 2*border_width
    new_h = with_tb.shape[0]
    left_border = np.tile(black, (new_h, border_width, 1))
    right_border = np.tile(black, (new_h, border_width, 1))
    # 拼接左右边框
    with_border = np.concatenate([left_border, with_tb, right_border], axis=1)
    
    return with_border


# 手动定义7x7圆形核（1参与计算，0不参与），可自行修改为5x5/9x9
circular_kernel = torch.tensor([
    [0, 1,  1,  1, 0],
    [1, 1,  1,  1, 1],
    [1, 1,  1,  1, 1],
    [1, 1,  1,  1, 1],
    [0, 1,  1,  1, 0]
], dtype=torch.float32)

def circular_max_dilation(dep):
    """
    极简版圆形最大值膨胀，修复维度不匹配问题
    Args:
        dep: 输入深度图，支持 (H,W)/(C,H,W)/(B,C,H,W) 格式
    Returns:
        膨胀后的深度图，形状与输入一致
    """
    # 保存原始形状，用于最后还原
    original_shape = dep.shape
    # 统一转为4维 (B, C, H, W)
    if len(original_shape) == 2:
        dep = dep.unsqueeze(0).unsqueeze(0)  # (H,W) → (1,1,H,W)
    elif len(original_shape) == 3:
        dep = dep.unsqueeze(0)  # (C,H,W) → (1,C,H,W)
    
    B, C, H, W = dep.shape
    kernel_size = circular_kernel.shape[0]
    padding = kernel_size // 2
    
    # 1. 填充边界（用-inf避免0干扰最大值计算）
    padded_dep = F.pad(dep, (padding, padding, padding, padding), value=-float('inf'))
    
    # 2. 展开滑动窗口：输出形状 (B, C*kernel_size², H*W)
    unfolded = F.unfold(padded_dep, kernel_size=kernel_size, stride=1)
    # 重塑为 (B, C, kernel_size², H, W)
    unfolded = unfolded.view(B, C, kernel_size*kernel_size, H, W)
    
    # 3. 生成匹配维度的掩码（核心修复：让掩码广播到H×W维度）
    # 展平核并扩展维度：(kernel_size²,) → (1, 1, kernel_size², 1, 1)
    kernel_flat = circular_kernel.flatten()
    mask = kernel_flat.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # 掩码广播到与unfolded相同的形状（自动匹配H×W）
    mask = mask.expand(B, C, -1, H, W)
    
    # 4. 应用掩码：将掩码外的像素设为-inf，不参与最大值计算
    unfolded[mask == 0] = -float('inf')
    
    # 5. 计算每个窗口的最大值，还原形状
    dilated = unfolded.max(dim=2)[0]  # (B,C,k²,H,W) → (B,C,H,W)
    
    # 6. 还原为原始输入形状
    if len(original_shape) == 2:
        dilated = dilated.squeeze(0).squeeze(0)  # 回到 (H,W)
    elif len(original_shape) == 3:
        dilated = dilated.squeeze(0)  # 回到 (C,H,W)
    
    return dilated

from icecream import ic
def summary(sample, output, path_output, setting):
    img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    # img_mean = torch.tensor((90.995, 96.2278, 94.3213)).view(1, 3, 1, 1)
    # img_std = torch.tensor((79.2382, 80.5267, 82.1483)).view(1, 3, 1, 1)
    # img_mean, img_std = (90.995, 96.2278, 94.3213), (79.2382, 80.5267, 82.1483)
    with torch.no_grad():
        if setting.spn_enable == True:
            _, _, H, W = output['pred_init'].shape
            feat_init = output['pred_init']
            list_feat = output['list_feat']
            # offset = output['offset']
            # aff = output['aff']
            rgb = sample['rgb'].detach()
            # rgb.mul_( img_std.type_as(rgb)).add_( img_mean.type_as(rgb))
            dep = sample['dep'].detach()
            # dep = torch.max_pool2d(dep,3,1,1)
            dep = circular_max_dilation(dep)
            pred = output['pred'].detach()
            gt = sample['gt'].detach()
            pred = torch.clamp(pred, min=0)

            rgb = rgb[0, :, :, :].data.cpu().numpy()
            dep = dep[0, 0, :, :].data.cpu().numpy()
            pred = pred[0, 0, :, :].data.cpu().numpy()
            gt = gt[0, 0, :, :].data.cpu().numpy()
            feat_init = feat_init[0, 0, :, :].data.cpu().numpy()

            rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
            # dep_max = dep.max()
            # dep = dep / dep_max
            # pred = pred / dep_max
            # pred_gray = pred
            # gt = gt / dep_max
            # feat_init = feat_init / dep_max


            norm_dep = plt.Normalize(vmin=dep.min(), vmax=dep.max())
            norm_gt = plt.Normalize(vmin=gt.min(), vmax=gt.max())
            norm_pred = plt.Normalize(vmin=pred.min(), vmax=pred.max())
            

            """临时添加，为了统一normlize和可视化"""
            norm_ = plt.Normalize(vmin=0, vmax=gt.max())
            norm_gt = plt.Normalize(vmin=gt.min(), vmax=gt.max())
            norm_pred = plt.Normalize(vmin=0, vmax=gt.max()*1.0)
            # print(gt.max())
            # norm_pred = plt.Normalize(vmin=0, vmax=4.8)
        
            rgb_np = np.clip(rgb, 0, 256).astype('uint8')
            # dep = (255.0 * cm(dep)).astype('uint8')
            # pred = (255.0 * cm(pred)).astype('uint8')
            # gt = (255.0 * cm(gt)).astype('uint8')
            # feat_init = (255.0 * cm(feat_init)).astype('uint8')

            rgb = Image.fromarray(rgb_np, 'RGB')
            # dep = Image.fromarray(dep[:, :, :3], 'RGB')
            # pred = Image.fromarray(pred[:, :, :3], 'RGB')
            # pred_gray = Image.fromarray(pred_gray)
            # gt = Image.fromarray(gt[:, :, :3], 'RGB')
            # feat_init = Image.fromarray(feat_init[:, :, :3], 'RGB')

            # for k in range(0, len(list_feat)):
            #     feat_inter = list_feat[k]
            #     feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
            #     feat_inter = feat_inter / dep_max
            #     feat_inter = (255.0 * cm(feat_inter)).astype('uint8')
            #     feat_inter = Image.fromarray(feat_inter[:, :, :3], 'RGB')

            #     list_feat[k] = feat_inter

            tem_FB_dep_greys = ( cm_tmp(norm_pred(dep)))
            tem_FB_dep_greys_valid_map = (tem_FB_dep_greys[:,:,1]==1)
            dep = (cm_jet(norm_pred(dep)))

            dep[tem_FB_dep_greys_valid_map,:] = 1
            # dep = add_2d_black_border(dep, border_width=2)

            pred = (cm_jet(norm_pred(pred)))
            gt = (cm_jet(norm_pred(gt)))
            feat_init = (cm_jet(norm_pred(feat_init)))

            burn_vislizeS =[]
            for k in range(0, len(list_feat)):
                feat_inter = list_feat[k]
                feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()

                # if k>2:
                #     #专门为了可视化改的
                #     # ic(feat_inter.min(), feat_inter.max())
                #     norm_pred = plt.Normalize(vmin=feat_inter.min(), vmax=feat_inter.max())

                tem_feat_inter_greys = ( cm_tmp(norm_pred(feat_inter)))
                tem_feat_inter_greys_valid_map = (tem_feat_inter_greys[:,:,1]==1)
                feat_inter = (cm_jet(norm_pred(feat_inter)))
                feat_inter[tem_feat_inter_greys_valid_map,:] = 1

                
                # """画pipeline用"""
                # FIG = render_depth_with_burn_upscaled(feat_inter, index = k, dep_rgba_full = dep, side_orig =102)
                # burn_vislizeS.append(FIG)

                list_feat[k] = feat_inter

            path_save_rgb = '{}/01_rgb.png'.format(path_output)
            path_save_dep = '{}/02_dep.png'.format(path_output)
            path_save_init = '{}/03_pred_init.png'.format(path_output)
            path_save_pred = '{}/05_pred_final.png'.format(path_output)
            path_save_gt = '{}/06_gt.png'.format(path_output)

            rgb.save(path_save_rgb)
            # dep.save(path_save_dep)
            # pred.save(path_save_pred)
            # feat_init.save(path_save_init)
            # gt.save(path_save_gt)

            
            
            plt.imsave(path_save_dep, dep)
            plt.imsave(path_save_pred, pred)
            plt.imsave(path_save_init, feat_init)
            plt.imsave(path_save_gt, gt)

            for k in range(0, len(list_feat)):
                path_save_inter = '{}/04_pred_prop_{:02d}.png'.format(path_output, k)
                # list_feat[k].save(path_save_inter)
                plt.imsave(path_save_inter, list_feat[k])

            # """画pipeline用"""
            # for k in range(0, len(burn_vislizeS)):
            #     path_save_inter = '{}/04_1125TEMP_burn_vislizeS_{:02d}.png'.format(path_output, k)
            #     burn_vislizeS[k].savefig(path_save_inter, dpi=140, bbox_inches='tight', pad_inches=0)
            

            try:
                prefilledslist_feat = output['prefilleds']
                for k in range(0, len(prefilledslist_feat)):
                    feat_inter = prefilledslist_feat[k]
                    feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
                    feat_inter = (cm_jet(norm_pred(feat_inter)))
                    prefilledslist_feat[k] = feat_inter

                    path_save_inter = '{}/7777_prefilleds_{:02d}.png'.format(path_output, k)
                    # list_feat[k].save(path_save_inter)
                    plt.imsave(path_save_inter, prefilledslist_feat[k])
            except:
                pass
            
                
            # try:
            #     y_before_inter = output['y_before_inter']

            #     single_prop_time = settings.prop_time//4

            #     current_gts = []
            #     gt = sample['gt'].detach()
            #     for k in range(0, len(y_before_inter)):
            #         if k <single_prop_time:
            #                 C = (gt > 0).float()
            #                 gt_eight = F.avg_pool2d(gt, kernel_size = 8, stride = 8, padding=(2,0)) / (F.avg_pool2d(C, kernel_size = 8, stride = 8, padding=(2,0)) + 0.0001)
            #                 current_gts.append(gt_eight)
            #         else:
            #             stage = int(k//single_prop_time)
            #             if stage==1:
            #                 downsample_ratio = 4
            #             elif stage ==2:
            #                 downsample_ratio = 2
            #             elif stage ==3:
            #                 downsample_ratio = 1
            #             current_gts.append(down_sample(gt, downsample_ratio))

            #     for k in range(0, len(y_before_inter)):
                    
            #         err = get_error_map(current_gts[k], y_before_inter[k])
            #         err_numpy = err[0, 0, :, :].data.cpu().numpy()
            #         norm_err = plt.Normalize(vmin=0, vmax=err_numpy.max())
            #         err_ = (cm(norm_err(err_numpy)))
            #         path_save_inter = '{}/06_errmap_before_Prop_{:02d}.png'.format(path_output, k)
            #         plt.imsave(path_save_inter, err_)


            #         feat_inter = y_before_inter[k]
            #         feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()

            #         feat_inter = (cm(norm_pred(feat_inter)))
            #         y_before_inter[k] = feat_inter

            #         path_save_inter = '{}/05_before_prop_{:02d}.png'.format(path_output, k)
            #         plt.imsave(path_save_inter, y_before_inter[k])
                    
            # except Exception as e:
            #     # 如果发生了其他类型的异常，也打印出错误信息和相关的图像对象信息
            #     pass
            
            try:
                DA = sample['DA'][0, 0, :, :].data.cpu().numpy()
                norm_pred = plt.Normalize(vmin=0, vmax=DA.max())
                DA = (cm_jet(norm_pred(DA)))
                
                path_save_gt = '{}/06_DAgt.png'.format(path_output)
                plt.imsave(path_save_gt, DA)

            except:
                pass


            try:
                confis_list = output['confis']
                norm_confi = plt.Normalize(vmin=0, vmax=2)
                for k in range(0, len(confis_list)):
                    confi = confis_list[k]
                    for confi_toward in range(confi.size(1)):
                        confi_specify_toward = confi[0, confi_toward, :, :].data.cpu().numpy()
                        confi_specify_toward = (cm_jet(norm_confi(confi_specify_toward)))
                        path_ = '{}/777_confi{:01d}_toward{:02d}.png'.format(path_output, int(3-k), confi_toward)
                        # list_feat[k].save(path_save_inter)
                        plt.imsave(path_, confi_specify_toward)
            except:
                pass

            try:
                ideal_strengths_list = output['ideal_strengths'][:6]
                
                for k in range(0, len(ideal_strengths_list)):
                    ideal_strength = ideal_strengths_list[k][0].data.cpu().numpy()
                    max = ideal_strength.max()*1.4
                    norm_strenth = plt.Normalize(vmin=0, vmax=max)
                    ideal_strength_np = (cm_jet(norm_strenth(ideal_strength)))
                    path_ = '{}/773_strgenth{:01d}.png'.format(path_output, int(k))
                    # list_feat[k].save(path_save_inter)
                    plt.imsave(path_, ideal_strength_np)

                split_masks_list = output['ideal_strengths'][6:]
                # 定义三种颜色：0 → 白, 1 → 黄, 999 → 红
                cmap = ListedColormap(['#FFFFFF', '#F7C846', '#F07848'])
                norm = BoundaryNorm([-0.5, 0.5, 1.5, 1000], cmap.N)
                for k in range(0, len(split_masks_list)):
                    split_mask = split_masks_list[k][0].data.cpu().numpy()
                    
                    path_ = '{}/773_strgenth{:01d}.png'.format(path_output, int(k))
                    # 绘制并保存（无显示）
                    plt.imshow(split_mask, cmap=cmap, norm=norm)
                    plt.axis('off')
                    plt.savefig(path_, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
            except:
                pass


            try:
                Dlist_feat = output['D_list_feat']
                for k in range(0, len(Dlist_feat)):
                    feat_inter = Dlist_feat[k]
                    feat_inter = feat_inter[0, 0, :, :].data.cpu().numpy()
                    feat_inter = (cm(norm_pred(feat_inter)))
                    Dlist_feat[k] = feat_inter

                    path_save_inter = '{}/08_D_pred_prop_{:02d}.png'.format(path_output, k)
                    # list_feat[k].save(path_save_inter)
                    plt.imsave(path_save_inter, Dlist_feat[k])
            except:
                pass

            try:
                annular_maskas =  output['annular_mask']

                temp_annular_mask = annular_maskas[0]
                temp_annular_mask = F.interpolate(temp_annular_mask, size=(228, 304), mode="bilinear")
                annular_rgb = temp_annular_mask * sample['rgb'].detach()
                annular_rgb = annular_rgb[0, :, :, :].data.cpu().numpy()
                
                annular_rgb = 255.0 * np.transpose(annular_rgb, (1, 2, 0))
                annular_rgb = np.clip(annular_rgb, 0, 256).astype('uint8')
                annular_rgb = Image.fromarray(annular_rgb, 'RGB')
                annular_rgb.save('{}/099_rgb.png'.format(path_output))

                norm_M = plt.Normalize(vmin=0, vmax=1)
                for k in range(0, len(annular_maskas)):
                    annular_mask = annular_maskas[k]
                    annular_mask = annular_mask[0, 0, :, :].detach().data.cpu().numpy()
                    annular_mask = (cm(norm_M(annular_mask)))
                    annular_maskas[k] = annular_mask

                    path_save_inter = '{}/07_annular_mask_{:02d}.png'.format(path_output, k)
                    plt.imsave(path_save_inter, annular_maskas[k])
            except:
                pass

            if setting.spn_module == "deform_dyspn":
                offset2 = output['offset2']
                x = np.array([])
                y = np.array([])
                offset_t = offset.cpu().numpy()
                for k in range(3):
                    for j in range(3):
                        if k != 1 & j != 1:
                            x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                            y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                x2 = np.array([])
                y2 = np.array([])
                offset_t2 = offset2.cpu().numpy()
                for k in range(3):
                    for j in range(3):
                        if k != 1 & j != 1:
                            x2 = np.hstack((x2, (offset_t2[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                            y2 = np.hstack((y2, (offset_t2[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                min_x = np.min((np.min(x), np.min(x2)) )
                max_x = np.max((np.max(x), np.max(x2)) )
                min_y = np.min((np.min(y), np.min(y2)) )
                max_y = np.max((np.max(y), np.max(y2)) )
                h = plt.hist2d(x, y, bins=300, cmap='jet', norm=colors.LogNorm(),
                               range=[[min_x, max_x], [min_y, max_y]]
                               # ,weights=list_w[i]
                               )
                cbar = plt.colorbar(h[3])
                density = cbar.get_ticks()
                plt.clim(density.min(), density.max())
                plt.savefig('{}/07_offset.png'.format(path_output))
                plt.clf()
                plt.cla()

                h = plt.hist2d(x2, y2, bins=300, cmap='jet', norm=colors.LogNorm(),
                               range=[[min_x, max_x], [min_y, max_y]]
                               # ,weights=list_w[i]
                               )
                cbar = plt.colorbar(h[3])
                density = cbar.get_ticks()
                plt.clim(density.min(), density.max())
                plt.savefig('{}/07_offset2.png'.format(path_output))
                plt.clf()
                plt.cla()
                dynamic = output['dynamic'].cpu().numpy()[0,...]
                dynamic_list = np.array_split(dynamic, 24, axis=0)
                image_list = []
                for i in range(6):
                    image = np.concatenate(dynamic_list[i*4:(i+1)*4][::-1], axis=1)
                    image_list.append(image)
                image = np.concatenate(image_list,axis=2)[0,:,:]
                image = (255.0 * cm2(image)).astype('uint8')
                path_save_dynamic = '{}/04_dynamic.png'.format(path_output)
                image = Image.fromarray(image[:, :, :3], 'RGB')
                image.save(path_save_dynamic)
            elif setting.spn_module=="dyspn":
                list_x = []
                list_y = []
                # list_w = []
                for i in range(len(offset)):
                    x = np.array([])
                    y = np.array([])
                    w = np.array([])
                    offset_t = offset[i].cpu().numpy()
                    # aff_t = aff[i].cpu().numpy()
                    # dysamplev6
                    # x = np.hstack((x, (offset_t[:,0,:,:]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[:,1,:,:]).reshape((-1))))
                    # dysamplev7
                    # x = np.hstack((x, (offset_t[..., 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[..., 1]).reshape((-1))))
                    # dysamplev8
                    x = np.hstack((x, (offset_t[0, :, :, :, 0]).reshape((-1))))
                    y = np.hstack((y, (offset_t[0, :, :, :, 1]).reshape((-1))))
                    x = np.hstack((x, (offset_t[..., 0]).reshape((-1)))) * W / 2
                    y = np.hstack((y, (offset_t[..., 1]).reshape((-1)))) * H / 2
                    # for k in range(3):
                    #     for j in range(3):
                    #         x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                    #         y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                    # w = np.hstack((w, (aff_t[0, 3 * k + j, :, :]).reshape((-1))))
                    list_x.append(x)
                    list_y.append(y)
                    # list_w.append(w)
                if len(offset) > 0:
                    min_x = np.min(np.concatenate(list_x))
                    max_x = np.max(np.concatenate(list_x))
                    min_y = np.min(np.concatenate(list_y))
                    max_y = np.max(np.concatenate(list_y))
                for i in range(len(offset)):
                    h = plt.hist2d(list_x[i], list_y[i], bins=300, cmap='jet', norm=colors.LogNorm(),
                                   range=[[min_x, max_x], [min_y, max_y]]
                                   # ,weights=list_w[i]
                                   )
                    cbar = plt.colorbar(h[3])
                    density = cbar.get_ticks()
                    plt.clim(density.min(), density.max())
                    plt.savefig('{}/07_offset_{:02d}.png'.format(path_output, i))
                    plt.clf()
                    plt.cla()
                list_x = []
                list_y = []
                # list_w = []
                ref_y = torch.linspace(-H + 1, H - 1, H, device=torch.device("cpu"))
                ref_x = torch.linspace(-W + 1, W - 1, W, device=torch.device("cpu"))
                for i in range(len(offset)):
                    x = np.array([])
                    y = np.array([])
                    w = np.array([])

                    offset_t = offset[i].cpu()
                    offset_t[..., 0] = (offset_t[..., 0] * W - ref_x.view(1, 1, 1, W)) / 2
                    offset_t[..., 1] = (offset_t[..., 1] * H - ref_y.view(1, 1, H, 1)) / 2

                    # aff_t = aff[i].cpu().numpy()
                    # dysamplev7
                    # x = np.hstack((x, (offset_t[0, :, :, 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[0, :, :, 1] ).reshape((-1))))
                    # dysamplev8
                    # x = np.hstack((x, (offset_t[0, :, :, :, 0]).reshape((-1))))
                    # y = np.hstack((y, (offset_t[0, :, :, :, 1]).reshape((-1))))
                    x = np.hstack((x, (offset_t[..., 0].numpy()).reshape((-1))))
                    y = np.hstack((y, (offset_t[..., 1].numpy()).reshape((-1))))
                    # for k in range(3):
                    #     for j in range(3):
                    #         x = np.hstack((x, (offset_t[0, 2 * (3 * k + j), :, :] + j - 1).reshape((-1))))
                    #         y = np.hstack((y, (offset_t[0, 2 * (3 * k + j) + 1, :, :] + k - 1).reshape((-1))))
                    # w = np.hstack((w, (aff_t[0, 3 * k + j, :, :]).reshape((-1))))
                    list_x.append(x)
                    list_y.append(y)
                    # list_w.append(w)
                if len(offset) > 0:
                    min_x = np.min(np.concatenate(list_x))
                    max_x = np.max(np.concatenate(list_x))
                    min_y = np.min(np.concatenate(list_y))
                    max_y = np.max(np.concatenate(list_y))
                for i in range(len(offset)):
                    h = plt.hist2d(list_x[i], list_y[i], bins=300, cmap='jet', norm=colors.LogNorm(),
                                   range=[[min_x, max_x], [min_y, max_y]]
                                   # ,weights=list_w[i]
                                   )
                    cbar = plt.colorbar(h[3])
                    density = cbar.get_ticks()
                    plt.clim(density.min(), density.max())
                    plt.savefig('{}/07_offset2_{:02d}.png'.format(path_output, i))
                    plt.clf()
                    plt.cla()
        else:
            rgb = sample['rgb'].detach()
            dep = sample['dep'].detach()
            dep = torch.max_pool2d(dep,3,1,1)
            pred = output['pred'].detach()
            gt = sample['gt'].detach()
            pred = torch.clamp(pred, min=0)

            rgb = rgb[0, :, :, :].data.cpu().numpy()
            dep = dep[0, 0, :, :].data.cpu().numpy()
            pred = pred[0, 0, :, :].data.cpu().numpy()
            gt = gt[0, 0, :, :].data.cpu().numpy()

            rgb = 255.0 * np.transpose(rgb, (1, 2, 0))
            dep_max = dep.max()
            dep = dep / dep_max
            pred = pred / dep_max
            pred_gray = pred
            gt = gt / dep_max

            rgb = np.clip(rgb, 0, 256).astype('uint8')
            dep = (255.0 * cm(dep)).astype('uint8')
            pred = (255.0 * cm(pred)).astype('uint8')
            pred_gray = (255.0 * pred_gray).astype('uint8')
            gt = (255.0 * cm(gt)).astype('uint8')

            rgb = Image.fromarray(rgb, 'RGB')
            dep = Image.fromarray(dep[:, :, :3], 'RGB')
            pred = Image.fromarray(pred[:, :, :3], 'RGB')
            pred_gray = Image.fromarray(pred_gray)
            gt = Image.fromarray(gt[:, :, :3], 'RGB')



            path_save_rgb = '{}/01_rgb.png'.format(path_output)
            path_save_dep = '{}/02_dep.png'.format(path_output)
            path_save_pred = '{}/05_pred_final.png'.format(path_output)
            path_save_pred_gray = '{}/05_pred_final_gray.png'.format(path_output)
            path_save_gt = '{}/06_gt.png'.format(path_output)

            rgb.save(path_save_rgb)
            dep.save(path_save_dep)
            pred.save(path_save_pred)
            pred_gray.save(path_save_pred_gray)
            gt.save(path_save_gt)


def get_halfsparse_depth( dep):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.contiguous().view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_idx//3]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))

    dep_sp = dep * mask.type_as(dep)

    return dep_sp

def get_error_map(gt, depth_pred):
    assert depth_pred.size() == gt.size(), f"Size mismatch: predict.size()={depth_pred.size()}, gt.size()={gt.size()}"

    err = torch.abs(depth_pred - gt)
    return err

def down_sample(depth, stride):
    C = (depth > 0).float()
    output = F.avg_pool2d(depth, stride, stride) / (F.avg_pool2d(C, stride, stride) + 0.0001)
    return  output