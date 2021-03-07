# LWANet
This repository contains the codes for our paper: [Light-weight Network for Real-time Adaptive Stereo Depth Estimation](https://www.sciencedirect.com/science/article/pii/S0925231221002599) 

# Abstract
Self-supervised learning methods have been proved effective in the task of real-time stereo
depth estimation with the requirement of lower memory space and less computational cost. In this
paper, a light-weight adaptive network (LWANet) is proposed by combining the self-supervised
learning method to perform online adaptive stereo depth estimation for low computation cost and
low GPU memory space. Instead of a regular 3D convolution, the pseudo 3D convolution is
employed in the proposed light-weight network to aggregate the cost volume for achieving a better
balance between the accuracy and the computational cost. Moreover, based on U-Net architecture,
the downsample feature extractor is combined with a refined convolutional spatial propagation
network (CSPN) to further refine the estimation accuracy with little memory space and
computational cost. Extensive experiments demonstrate that the proposed LWANet effectively
alleviates the domain shift problem by online updating the neural network, which is suitable for
embedded devices such as NVIDIA Jetson TX2.


# Acknowledgement

Many thanks to authors of [AnyNet](https://github.com/mileyan/AnyNet), [CSPN](https://github.com/XinJCheng/CSPN) for open-sourcing the code.

# Citaton

