#coding=utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .cspn import  Affinity_Propagate
from .feature_extraction import F1, F2, F3,  F2_UP,  F3_UP , F1_UP
from .Aggregation_submodules import LWANet_Aggregation
from .cost import _build_cost_volume
from .loss import self_supervised_loss


class LWANet(nn.Module):
    def __init__(self, args):
        super(LWANet, self).__init__()

        #self.init_channels = args.init_channels
        self.maxdisplist = args.maxdisplist
        self.with_cspn = args.with_cspn
        self.model_types =args.model_types  # "LWANet: 3D orP3D
        self.conv_3d_types1 = args.conv_3d_types1
        self.conv_3d_types2 = args.conv_3d_types2
        self.cost_volume = args.cost_volume
        self.maxdisp = args.maxdisp


        self.F1 = F1()
        self.F2 = F2()
        self.F3 = F3()

        self.F1_CSPN = F1()
        self.F2_CSPN = F2()
        self.F3_CSPN = F3()

        if self.cost_volume =="Distance_based":

            self.volume_postprocess = LWANet_Aggregation( input_planes=1, planes=8,
                                                          conv_3d_types1 = self.conv_3d_types1,
                                                          conv_3d_types2 = self.conv_3d_types2)

        elif self.cost_volume =="Difference":
            self.volume_postprocess = LWANet_Aggregation(input_planes=16, planes=12,
                                                         conv_3d_types1=self.conv_3d_types1,
                                                         conv_3d_types2=self.conv_3d_types2)

        elif self.cost_volume =="Concat":
            self.volume_postprocess = LWANet_Aggregation(input_planes=32, planes=12,
                                                         conv_3d_types1=self.conv_3d_types1,
                                                         conv_3d_types2=self.conv_3d_types2)

        if self.with_cspn:

            self.F2_UP = F2_UP()
            self.F3_UP = F3_UP()
            self.F1_UP = F1_UP()

            cspn_config_default = {'step':4, 'kernel': 3, 'norm_type': '8sum'}
            self.post_process_layer = [self._make_post_process_layer(cspn_config_default)]
            self.post_process_layer = nn.ModuleList(self.post_process_layer)

        self.self_supervised_loss = self_supervised_loss(n=1, SSIM_w=0.85, disp_gradient_w=0.1, lr_w=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def _make_post_process_layer(self, cspn_config=None):
        return Affinity_Propagate(cspn_config['step'],
                                               cspn_config['kernel'],
                                               norm_type=cspn_config['norm_type'])

    def forward(self, left, right):


        img_size = left.size()

        feats_l_F1 = self.F1(left)

        feats_l_F2 = self.F2(feats_l_F1)

        feats_l_F3 = self.F3(feats_l_F2)


        feats_l = feats_l_F3


        feats_r_F1 = self.F1(right)

        feats_r_F2 = self.F2(feats_r_F1)

        feats_r_F3 = self.F3(feats_r_F2)

        feats_r = feats_r_F3


        pred = []

        cost = _build_cost_volume(self.cost_volume, feats_l, feats_r, self.maxdisp)

        cost = self.volume_postprocess(cost).squeeze(1)


        pred_low_res_left = disparityregression2(0, self.maxdisplist[0])(F.softmax(-cost, dim=1))

        pred_low_res = pred_low_res_left * img_size[2] / pred_low_res_left.size(2)

        disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')

        pred.append(disp_up)


        if self.with_cspn:

            feats_l_F1_CSPN = self.F1_CSPN(left)

            feats_l_F2_CSPN = self.F2_CSPN(feats_l_F1_CSPN)

            feats_l_F3_CSPN = self.F3_CSPN(feats_l_F2_CSPN)



            F3_UP = torch.cat((self.F3_UP(feats_l_F3_CSPN), feats_l_F2_CSPN), 1)

            F2_UP = torch.cat((self.F2_UP(F3_UP), feats_l_F1_CSPN), 1)

            F1_UP = self.F1_UP(F2_UP)

            x = self.post_process_layer[0](F1_UP, disp_up)

            pred.append(x)

        loss = []


        if self.train:
             for outputs in pred:
                loss.append(self.self_supervised_loss(outputs, [left, right]))

        else:
            loss = [0]

        pred = [torch.squeeze(pred, 1) for pred in pred]

        return pred, loss



class disparityregression2(nn.Module):
    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = Variable(torch.arange(start*stride, end*stride, stride).view(1, -1, 1, 1).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        disp = disp.float()

        out = torch.sum(x*disp, 1, keepdim=True)
        return out



class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()



def apply_disparity(img, disp):

    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                           padding_mode='zeros')

    return output



def generate_image_left( img, disp):
    return apply_disparity(img, -disp)


