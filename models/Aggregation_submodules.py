#coding=utf-8
from __future__ import print_function
import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def activation_function(types = "ELU"):     # ELU or Relu


    if types == "ELU":

        return nn.Sequential(nn.ELU(inplace=True))

    elif types == "Mish":

        nn.Sequential(Mish())

    elif types == "Relu":

        return nn.Sequential(nn.ReLU(inplace=True))

    else:

        AssertionError("please define the activate function types")




class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * torch.tanh(F.softplus(input))



# cost aggregation submodule

def conv_3d(in_planes, out_planes, kernel_size, stride, pad, conv_3d_types="3D"):


    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)
            )


    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),

            )


    else:

        AssertionError("please define conv_3d_types")




def convbn_3d(in_planes, out_planes, kernel_size, stride, pad, conv_3d_types="3D"):


    if conv_3d_types == "3D":

        return nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))


    elif conv_3d_types == "P3D":  # 3*3*3　to 1*3*3 + 3*1*1

        return nn.Sequential(

            nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),

            nn.BatchNorm3d(out_planes))


    else:

        AssertionError("please define conv_3d_types")



def convTranspose3d(in_planes, out_planes, kernel_size, stride, padding=1, conv_3d_types="P3D"):

    if conv_3d_types == '3D':
        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size, padding = padding, output_padding=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_planes))


    elif conv_3d_types == "P3D":

        return nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size, padding=padding, output_padding=1, stride=stride,
                               bias=False),
            nn.BatchNorm3d(out_planes))


    else:
        AssertionError("please define conv_3d_types")




class LWANet_Aggregation(nn.Module):  # base on PSMNet basic
    def __init__(self, input_planes=8, planes=16, maxdisp=192,  conv_3d_types1 = "P3D", conv_3d_types2 = "P3D", activation_types2 = "ELU"):
        super(LWANet_Aggregation, self).__init__()
        self.maxdisp = maxdisp

        self.pre_3D = nn.Sequential(
                            convbn_3d(input_planes, planes, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                            activation_function(types = activation_types2),
                            convbn_3d(planes, planes, 3, 2, 1, conv_3d_types =  conv_3d_types1),
                            activation_function(types = activation_types2)
                                    )

        self.middle_3D = nn.Sequential(

                        convbn_3d(planes, planes*2, 3, 1, 1,  conv_3d_types =  conv_3d_types2),
                        activation_function(types = activation_types2),
                        convbn_3d(planes*2, planes*4, 3, 1, 1, conv_3d_types =  conv_3d_types2),
                        activation_function(types = activation_types2),
                        convbn_3d(planes * 4, planes * 4, 3, 1, 1, conv_3d_types=conv_3d_types2),
                        activation_function(types=activation_types2),
                        convbn_3d(planes * 4, planes * 2, 3, 1, 1, conv_3d_types=conv_3d_types2),
                        activation_function(types=activation_types2),
                        convTranspose3d(planes * 2, planes * 2, kernel_size=3, stride=2, conv_3d_types=conv_3d_types2),
                        activation_function(types=activation_types2)
                                    )

        self.post_3D = nn.Sequential(
                                convbn_3d(planes*2, planes, 3, 1, 1, conv_3d_types =  conv_3d_types1),
                                activation_function(types = activation_types2),
                                conv_3d(planes, 1, kernel_size=3, pad=1, stride=1, conv_3d_types =  conv_3d_types1)
                                    )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, cost):

        cost = self.pre_3D(cost)
        cost = self.middle_3D(cost)
        cost = self.post_3D(cost)


        return cost




