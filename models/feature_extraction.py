#coding=utf-8
from __future__ import print_function
import torch.nn as nn


class F1(nn.Module):
    def __init__(self):
        super(F1, self).__init__()
        # feature extraction
        self.init_feature = nn.Sequential(

            # 6-24
            nn.Conv2d(3, 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ELU(inplace=True),
            nn.Conv2d(4, 4, 3, 2, 1, bias=False),
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),

        )

    def forward(self, x_left):

        buffer_left = self.init_feature(x_left)

        return buffer_left



class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()

        self.init_feature = nn.Sequential(


            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),
            nn.ELU(inplace=True),

            nn.Conv2d(8, 12, 3, 1, 1, bias=False),
            nn.BatchNorm2d(12),
            nn.ELU(inplace=True),
            nn.Conv2d(12, 12, 3, 1, 1, bias=False),

        )

    def forward(self, x_left):

        buffer_left = self.init_feature(x_left)

        return buffer_left


class F3(nn.Module):
    def __init__(self):
        super(F3, self).__init__()

        self.init_feature = nn.Sequential(

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(12),
            nn.ELU(inplace=True),

            nn.Conv2d(12, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),

        )

    def forward(self, x_left):

        buffer_left = self.init_feature(x_left)

        return buffer_left




class F3_UP(nn.Module):
    def __init__(self):
        super(F3_UP, self).__init__()
        self.init_feature = nn.Sequential(

            nn.Conv2d(16, 16, 3, 1, 1, bias=False),

            nn.BatchNorm2d(16),

            nn.ELU(inplace=True),

            nn.ConvTranspose2d(16, 12, 3, 2, 1, output_padding=1, bias=False),
        )

    def forward(self, x_left):

        buffer_left = self.init_feature(x_left)

        return buffer_left


class F2_UP(nn.Module):
    def __init__(self):
        super(F2_UP, self).__init__()

        # cat
        self.init_feature = nn.Sequential(

            nn.BatchNorm2d(24),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(24, 8, 3, 2, 1, output_padding=1, bias=False),
        )



    def forward(self, x_left):
        ### feature extraction
        buffer_left = self.init_feature(x_left)

        return buffer_left


class F1_UP(nn.Module):
    def __init__(self):
        super(F1_UP, self).__init__()
        # cat
        self.init_feature = nn.Sequential(

            nn.BatchNorm2d(16),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1, bias=False),

        )


    def forward(self, x_left):
        ### feature extraction
        buffer_left = self.init_feature(x_left)

        return buffer_left



