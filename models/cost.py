import torch
import torch.nn as nn
from torch.autograd import Variable


def _build_volume_2d_anynet(feat_l, feat_r, maxdisp, stride=1):

    assert maxdisp % stride == 0  # Assume maxdisp is multiple of stride
    cost = torch.zeros((feat_l.size()[0], maxdisp//stride, feat_l.size()[2], feat_l.size()[3]), device='cuda')
    for i in range(0, maxdisp, stride):
        cost[:, i// stride, :, :i] = feat_l[:, :, :, :i].abs().sum(1)

        if i > 0:
            cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], 1, 1)
        else:
            cost[:, i // stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], 1, 1)

    return cost.contiguous()


def _build_volume_2d3_anynet( feat_l, feat_r, disp, maxdisp=3, stride=1):
    size = feat_l.size()
    batch_disp = disp[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, size[-2], size[-1])
    batch_shift = torch.arange(-maxdisp + 1, maxdisp, device='cuda').repeat(size[0])[:, None, None, None] * stride
    batch_disp = batch_disp - batch_shift.float()
    batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2], size[-1])
    batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2], size[-1])

    cost = torch.norm(batch_feat_l - warp(batch_feat_r, batch_disp), 1, 1)

    cost = cost.view(size[0], -1, size[2], size[3])

    return cost.contiguous()


def _build_volume_2d_psmnet( refimg_fea, targetimg_fea, maxdisp):
    cost = Variable(
        torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, maxdisp , refimg_fea.size()[2],
                          refimg_fea.size()[3]).zero_()).cuda()

    for i in range(maxdisp):
        if i > 0:

            cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
            cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
            cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea

    return cost.contiguous()




def _build_volume_2d3_psmnet(feat_l, feat_r, disp, maxdisp=3, stride=1):
    size = feat_l.size()

    batch_disp = disp[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, size[-2], size[-1])
    batch_shift = torch.arange(-maxdisp + 1, maxdisp, device='cuda').repeat(size[0])[:, None, None, None] * stride
    batch_disp = batch_disp - batch_shift.float()
    batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2], size[-1])
    batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2],
                                                                                     size[-1])
    #cost = batch_feat_l - warp(batch_feat_r, batch_disp)
    cost = torch.cat((batch_feat_l , warp(batch_feat_r, batch_disp)),  1).contiguous()

    cost = cost.view(size[0], size[1]*2, -1, size[2], size[3])
    # print("cost size", cost.shape)

    return cost.contiguous()




def _build_volume_2d_aanet(refimg_fea, targetimg_fea, maxdisp):


    b, c, h, w = refimg_fea.size()
    cost_volume = refimg_fea.new_zeros(b, maxdisp, h, w)

    for i in range(maxdisp):
        if i > 0:
            cost_volume[:, i, :, i:] = (refimg_fea[:, :, :, i:] *
                                        targetimg_fea[:, :, :, :-i]).mean(dim=1)
        else:
            cost_volume[:, i, :, :] = (refimg_fea * targetimg_fea).mean(dim=1)

    return cost_volume.contiguous()





def _build_volume_2d_difference(feat_l, feat_r, maxdisp):

    b, c, h, w = feat_l.size()


    cost_volume = feat_l.new_zeros(b, c, maxdisp, h, w)

    for i in range(maxdisp):
        if i > 0:
            cost_volume[:, :, i, :, i:] = feat_l[:, :, :, i:] - feat_r[:, :, :, :-i]
        else:
            cost_volume[:, :, i, :, :] = feat_l - feat_r

    return cost_volume





def _build_volume_2d3_difference( feat_l, feat_r, disp, maxdisp=3, stride=1):
    size = feat_l.size()
    batch_disp = disp[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, size[-2], size[-1])
    batch_shift = torch.arange(-maxdisp + 1, maxdisp, device='cuda').repeat(size[0])[:, None, None, None] * stride
    batch_disp = batch_disp - batch_shift.float()
    batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2], size[-1])
    batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2], size[-1])

    # cost = torch.norm(batch_feat_l - warp(batch_feat_r, batch_disp), 1, 1)

    cost = batch_feat_l - warp(batch_feat_r, batch_disp)

    cost = cost.view(size[0], size[1], -1,  size[2], size[3])

    return cost.contiguous()




def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    #output = nn.functional.grid_sample(x, vgrid, align_corners=True )
    output = nn.functional.grid_sample(x, vgrid)
    return output



def _build_cost_volume(cost_volume_type, refimg_fea, targetimg_fea, maxdisp):
    if cost_volume_type == "Concat":

        cost = _build_volume_2d_psmnet(refimg_fea, targetimg_fea, maxdisp=maxdisp // 8)


    elif cost_volume_type == "Distance_based":

        cost = _build_volume_2d_anynet(refimg_fea, targetimg_fea, maxdisp // 8, stride=1)
        cost = torch.unsqueeze(cost, 1)


    elif cost_volume_type == "Difference":
        #print("build difference")
        cost = _build_volume_2d_difference(refimg_fea, targetimg_fea, maxdisp // 8)
        #print("cost size:", cost.shape)


    else:
        AssertionError("please define cost volume types")

    return cost





def _build_redidual_cost_volume(cost_volume_type, L2, R2, wflow, maxdisp):
    if cost_volume_type == "Concat":

        cost_residual = _build_volume_2d3_psmnet(L2, R2, wflow, maxdisp)

    elif cost_volume_type == "Distance_based":
        cost_residual = _build_volume_2d3_anynet(L2, R2, wflow, maxdisp)
        cost_residual = torch.unsqueeze(cost_residual, 1)


    elif cost_volume_type == "Difference":
        cost_residual = _build_volume_2d3_difference(L2, R2, wflow, maxdisp)
        # cost_residual = torch.unsqueeze(cost_residual, 1)

    else:
        AssertionError("please define cost volume types")

    return cost_residual
