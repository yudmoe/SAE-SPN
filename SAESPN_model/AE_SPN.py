import torch.nn as nn
import torch
import cv2
import torchvision
from SAESPN_model.common import conv_bn_relu, conv_shuffle_bn_relu, convt_bn_relu
from SAESPN_model.stodepth_lineardecay import se_resnet34_StoDepth_lineardecay, se_resnet18_StoDepth_lineardecay,se_resnet68_StoDepth_lineardecay
import numpy as np
import torch.nn.functional as F

from thop import profile
# 增加可读性
from thop import clever_format
# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)
# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)
# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # # # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map


class get_D_diff_layer(nn.Module):
    def __init__(self,prop_kernel):
        super(get_D_diff_layer, self).__init__()

        self.kernel_size = prop_kernel
        self.number_of_neighbor = self.kernel_size*self.kernel_size
        self.num_guide = self.number_of_neighbor-1

        shift_used_convKernel_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i ==2 and j ==2:
                    continue

                temp_kernel = torch.zeros(1,self.kernel_size, self.kernel_size)
                temp_kernel[0,i,j] = -1
                temp_kernel[0,2,2] = 1
                # current_K = nn.Parameter(torch.concat(current_list,0).unsqueeze(1))
                shift_used_convKernel_list.append(temp_kernel)

        self.shift_used_convKernel_list = shift_used_convKernel_list
        tem = torch.concat(shift_used_convKernel_list,0)
        self.shift_used_convK = nn.Parameter(tem.unsqueeze(1))
        self.shift_used_convK.requires_grad = False


    def forward(self, Depth):
        # with torch.no_grad():
        shifited_Ddiff = F.conv2d(Depth, self.shift_used_convK, bias=None, stride=1, padding=int((self.kernel_size-1)/2), dilation=1)
        return shifited_Ddiff


class mySPN_affinity_inorder(nn.Module):
    def __init__(self, prop_kernel,prop_time):
        super(mySPN_affinity_inorder, self).__init__()

        
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        self.number_of_neighbor = self.prop_kernel*self.prop_kernel
        self.affinity = 'mySPN_affinity_inorder'

        self.get_initialD_diff = get_D_diff_layer(prop_kernel)

        for params in self.get_initialD_diff.parameters():
            params.requires_grad = False

    def _normalize_guide(self, guide):
        abs_guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)  + 0.000001 # B 1 H W
        guide_sum = torch.sum(guide, dim=1).unsqueeze(1)+1e-6 # B 1 H W

        half1, half2 = torch.chunk(guide, 2, dim=1)

        half1 = half1/(abs_guide_sum+1e-6)
        half2 = half2/(abs_guide_sum+1e-6)
        center_guide = (1 - guide_sum/(abs_guide_sum+1e-6))
        normalized_guide = torch.cat([half1,center_guide,half2],dim=1)
        return normalized_guide

    def _weight_guidance_byInitialD_and_norm(self,coarse_depth, guidance, weight, initialD_constant):
        initialD_diff = self.get_initialD_diff(coarse_depth)
        abs_initialD_diff = torch.abs(initialD_diff) +1e-6
        initialD_guide = torch.exp(-(initialD_constant+1e-6)*abs_initialD_diff)

        initialD_AFF = self._normalize_guide(initialD_guide)
        affinity = self._normalize_guide(guidance)

        affinity = affinity.mul(weight )
        initialD_AFF = initialD_AFF.mul(1-weight +1e-6)
        weighted_guidance = affinity + initialD_AFF
        return weighted_guidance

    def forward(self, coarse_depth, guidances, confidences, sparse_depth=None,rgb=None):
        
        # coarse_depth : [B x 1 x H x W]
        # guidances : [B x 48 x H/4 x W/4, B x 48 x H/2 x W/2, B x 48 x H/1 x W/1]
        # confidence : [B x 48 x H x W]
        # weights : [B x self.args.prop_time/3 x H/4 x W/4, B x self.args.prop_time/3 x H/2 x W/2,B x self.args.prop_time/3 x H/1 x W/1]
        
        """
        分别用 1/4分辨率affinity,1/2分辨率affinity,1/1分辨率affinity各自传播:
                1/3 prop_time       1/3 prop_time     1/3prop_time    
        次数
        """
        down_sample = nn.AvgPool2d(2,stride=2)
        up_sample = nn.Upsample(scale_factor=2, mode='nearest')


        # Propagation 

        feat_result = coarse_depth

        list_feat = []
        
        feat_result = down_sample(down_sample(feat_result))
        for i in range(3):
            current_guidances = guidances[i]
            current_weights = confidences[i]
            if i==1:
                feat_result = nn.Upsample(scale_factor=2, mode='nearest')(feat_result)
            if i==2:
                feat_result = nn.Upsample(scale_factor=2, mode='nearest')(feat_result)

            iteral_time = int(self.prop_time/3)
            for j in range(iteral_time):
                const_index = i*iteral_time + j
                bs, _, h, w = feat_result.size() 
                normalized_affinity_imgsize = self._weight_guidance_byInitialD_and_norm(feat_result, current_guidances,  
                                                                                        current_weights[:,j:j+1], 
                                                                                        self.affweight_scale_const[const_index])
                B, channel, H, W = normalized_affinity_imgsize.shape
                current_aff = normalized_affinity_imgsize.reshape(B, self.number_of_neighbor , H * W)

                depth_im2col = F.unfold(feat_result, self.prop_kernel, 1, int((self.prop_kernel-1)/2), 1)
                guide_result = torch.einsum('ijk,ijk->ik', (depth_im2col, current_aff))
                propageted_depth = guide_result.view(bs, 1, h, w)

                feat_result = propageted_depth
                list_feat.append(propageted_depth)

        return feat_result, list_feat , normalized_affinity_imgsize

    def _propagation_onece(self, current_D, guidances, weight, initialD_constant):
        bs, _, h, w = current_D.size() 
        normalized_affinity_imgsize = self._weight_guidance_byInitialD_and_norm(current_D, guidances,  weight, initialD_constant)
        B, channel, H, W = normalized_affinity_imgsize.shape
        current_aff = normalized_affinity_imgsize.reshape(B, self.number_of_neighbor , H * W)

        depth_im2col = F.unfold(current_D, self.prop_kernel, 1, int((self.prop_kernel-1)/2), 1)
        guide_result = torch.einsum('ijk,ijk->ik', (depth_im2col, current_aff))
        propageted_depth = guide_result.view(bs, 1, h, w)
        return propageted_depth, normalized_affinity_imgsize
    

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=(1, 1), bias=False)
class simpleBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(simpleBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


def adjust_depth_map(dep: torch.Tensor, bias = 0.5) -> torch.Tensor:
    """
    调整深度图：非0深度值减0.5，结果小于0则置0，0值保持不变
    Args:
        dep: 输入深度图，shape为[B, 1, H, W]，torch张量
    Returns:
        调整后的深度图，与输入shape和数据类型一致
    """
    # 1. 先对所有元素减bias
    dep_minus = dep - bias
    # 2. 用torch.clamp将小于0的部分截断为0
    dep_clamped = torch.clamp(dep_minus, min=0.0)
    # 3. 用torch.where保留原0值（仅对非0区域应用截断后的结果）
    # 掩码：dep != 0 标记有效深度区域
    mask = (dep != 0)
    adjusted_dep = torch.where(mask, dep_clamped, dep)
    
    return adjusted_dep

class HCSPN_Model(nn.Module):
    def __init__(self, prop_kernel, prop_time,data_name='NYU',norm_depth=[0.1, 10.0],  sto=True, res="res34", suffle_up=False, norm_layer=None):
        super(HCSPN_Model, self).__init__()
        self.data_name = data_name
        self.norm_depth = norm_depth
        self.prop_kernel = prop_kernel
        self.prop_time = prop_time
        self.num_guide = prop_kernel*prop_kernel -1 
        self.num_time_of_one_step = int(self.prop_time//1)
        self.half_k = int(self.prop_kernel//2)
         # Encoder
        self.conv1_rgb = conv_bn_relu(3, 36, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 14, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv_dep_initial = conv_bn_relu(1, 14, kernel=3, stride=1, padding=1,
                                      bn=False)
        if sto == True:
            if res == "res18":
                net = se_resnet18_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
            else:
                net = se_resnet34_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
                # net = se_resnet68_StoDepth_lineardecay(prob_0_L=[1.0, 0.5], pretrained=True)
        else:
            if res == "res18":
                net = torchvision.models.resnet18(pretrained=True)
            else:
                net = torchvision.models.resnet34(pretrained=True)
        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        if suffle_up == True:
            # 1/8
            self.dec5 = conv_shuffle_bn_relu(512, 256, kernel=3, stride=1, padding=1)
            # 1/4
            self.dec4 = conv_shuffle_bn_relu(256 + 512, 128, kernel=3, stride=1, padding=1)
            # 1/2
            self.dec3 = conv_shuffle_bn_relu(128 + 256, 64, kernel=3, stride=1, padding=1)
            # 1/
            self.dec2 = conv_shuffle_bn_relu(64 + 128, 64, kernel=3, stride=1, padding=1)
        else:
            # Shared Decoder
            # # 1/8
            self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2, padding=1, output_padding=1)

            # 1/4
            self.dec4 = convt_bn_relu(256 + 512 , 128, kernel=3, stride=2, padding=1, output_padding=1)


            # 1/2
            self.dec3 = convt_bn_relu(128+256, 128, kernel=3, stride=2,padding=1, output_padding=1)
            # 1/1
            self.dec2 = convt_bn_relu(128+128, 128, kernel=3, stride=2,
                                    padding=1, output_padding=1)
            
            self.gd0_dec_input1 = conv_bn_relu(128+64, 128, kernel=3, stride=1,padding=1)

            self.gd_dec0 = conv_bn_relu(64+64, self.num_guide, kernel=3, stride=1,
                                        padding=1, bn=False, relu=False)
            self.cf0_dec0 = nn.Sequential(
                    nn.Conv2d(64+64, self.num_time_of_one_step, kernel_size=3, stride=1, padding=1),
                )
            
            self.prop_layer = mySPN_affinity_inorder(prop_kernel = self.prop_kernel, prop_time = self.prop_time)
            #2024-4-19 D9v3版本把动态的initialDD的scale constant也融合进去了,每个scale在它的融合阶段都有不同的initialD的系数产生不一样的affinity
            #期望的scale constant是前面很大因为深度图很差后面很小。
            self.affweight_scale_const = nn.ParameterList([nn.Parameter(30* torch.ones(1)) for i in range(self.prop_time)])


    def _make_layer(self, block, out_channels, num_blocks):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = []
        for stride in range(num_blocks):
            layers.append(block( out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

    def _make_layer(self, block, out_channels, num_blocks):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        layers = []
        for stride in range(num_blocks):
            layers.append(block( out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    def down_sample(self, depth,stride):
        C = (depth > 0).float()
        output = F.avg_pool2d(depth, stride, stride) / (F.avg_pool2d(C, stride, stride) + 0.0001)
        return  output
    
    def forward(self, rgb, dep, intial_depths):
        
        #泛化性affinity调试
        dep = adjust_depth_map(dep,  0.0)
        intial_depths = adjust_depth_map(intial_depths,  0.0)

        bz = dep.shape[0]
        dep_max = torch.max(dep.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
        
        intial_depths = intial_depths
        intial_depths = intial_depths/(dep_max+1e-4)
        dep = dep/(dep_max+1e-4)

        # intial_depths3 = self.down_sample(intial_depths,8)
        y_inter=[]

        fe1_rgb = self.conv1_rgb(rgb)
        fe1_dep = self.conv1_dep(dep)
        fe_dep_initial = self.conv_dep_initial(intial_depths)
        fe_initial = torch.cat((fe1_rgb, fe1_dep, fe_dep_initial), dim=1)#36+14

        fe1 = fe_initial
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)#512 256 upsample 1/8
        eighth_feature_map = self._concat(fd5, fe5)


        fd4 = self.dec4(eighth_feature_map)#256+512 128 upsample 1/4
        quater_feature_map = self._concat(fd4, fe4)

        #1/2
        fd3 = self.dec3(quater_feature_map)#128+256+128 64 upsample 1/2 从1/4尺度到1/2尺度
        feature_initial1 =  self._concat(fd3, fe3)#两个1/2尺度的特征256到128
        half_feature_map = feature_initial1

        
        #1/1
        fd2 = self.dec2(half_feature_map)#64+128 64 upsample 1/1
        fd2 = self.gd0_dec_input1(torch.cat((fd2, fe2), dim=1)) #1/1 128
        layer0_output = fd2

        intial_depths = intial_depths
        current_depth = intial_depths
        guide0 = self.gd_dec0(layer0_output)
        conf0_beforeS = self.cf0_dec0(layer0_output)
        conf0 = nn.Sigmoid()(conf0_beforeS)
        # ic(conf0.mean(dim=[2, 3]), conf0.mean(dim=[1, 2, 3]))
        for index in range(self.num_time_of_one_step):
            constant_index =  index
            current_depth,  normalized_affinity_imgsize0 =  self.prop_layer._propagation_onece(current_depth, guide0, (conf0[:,index:index+1]), self.affweight_scale_const[constant_index])
            y_inter.append(current_depth)


        y = current_depth

        

        y_inter = [inter*dep_max for inter in y_inter]
        y = y * dep_max
        intial_depths = intial_depths * dep_max
        # Remove negative depth
        y = torch.clamp(y, min=self.norm_depth[0],max=self.norm_depth[1])

        return {'pred': y,
                'pred_init': intial_depths,
                "list_feat": y_inter,
                "offset": normalized_affinity_imgsize0,
                "aff": normalized_affinity_imgsize0
                }



import time
from icecream import ic
class Model(nn.Module):
    def __init__(self, data_name='NYU',iteration=3, num_neighbor=9, mode="dyspn", shuffle_up=False, norm_depth=[0.1, 10.0], res="res18",
                 bm="v1", norm_layer='bn', stodepth=True):
        super(Model, self).__init__()
        self.sto = stodepth
        # self.sto = False
        assert res in ["res18", "res34"]
        self.res = res
        self.bm = bm
        self.shuffle_up = shuffle_up
        self.mode = mode
        assert norm_layer in ['bn', 'in']
        self.norm_layer = norm_layer
        self.iteration = iteration
        self.num_sample = num_neighbor
        BM = HCSPN_Model
        self.base = BM(data_name=data_name, norm_depth=norm_depth, prop_time=self.iteration, prop_kernel=self.num_sample)
        
        self.norm = norm_depth

        self.cost_time_list = []
        self.allocated_memory_list=[]

    def forward(self, rgb0, dep, prefilled):

        # torch.cuda.synchronize()
        # t0 = time.time()

        otuput = self.base(rgb = rgb0, dep = dep, intial_depths = prefilled)

        # torch.cuda.synchronize()
        # self.cost_time_list.append(time.time()-t0)
        # avg_cost = sum(self.cost_time_list)/len(self.cost_time_list)
        # if len(self.cost_time_list)%100==0:
        #     ic(avg_cost)

        #     flops, params = profile(self.base, inputs=(rgb0, dep, prefilled,))
        #     flops, params = clever_format([flops, params], "%.3f")  
        #     ic(flops,   params )
        return otuput