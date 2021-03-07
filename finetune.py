import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
from dataloader import KITTIdatalist as ls
import utils.logger as logger
import torch.backends.cudnn as cudnn
from models.LWANet import *

import pdb

# 查看GPU使用情況
# watch --color -n1 gpustat -cpu


parser = argparse.ArgumentParser(description='LWANet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 3, 3])
parser.add_argument('--with_cspn', type =bool, default= True, help='with cspn network or not')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based"')
parser.add_argument('--lr', type=float, default=5e-4*0.5, help='learning rate')
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=8, help='batch size for training (default: 8)')
parser.add_argument('--test_bsize', type=int, default=8,help='batch size for testing (default: 8)')
parser.add_argument('--resume', type=str, default= None, help='resume path')
parser.add_argument('--print_freq', type=int, default=10, help='print frequence')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
parser.add_argument('--model_types', type=str, default='LWANet', help='model_types : 3D OR P3D')
parser.add_argument('--conv_3d_types1', type=str, default='P3D', help='model_types :  3D, P3D ')
parser.add_argument('--conv_3d_types2', type=str, default='P3D', help='model_types : 3D, P3D')


parser.add_argument('--save_path', type=str, default='/results/finetune2015/',help='the path of saving checkpoints and log')
parser.add_argument('--split_for_val', type =bool, default=False,  help='finetune for submission or for validation')
parser.add_argument('--datatype', default='mix', help='finetune dataset: 2012, 2015, mix')
parser.add_argument('--datapath2015', default='/data6/wsgan/KITTI/KITTI2015/training/', help='datapath')
parser.add_argument('--datapath2012', default='/data6/wsgan/KITTI/KITTI2012/training/', help='datapath')


args = parser.parse_args()


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    if args.datatype == '2015':

        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader2015(
            args.datapath2015, log, split=args.split_for_val)

    elif args.datatype == '2012':

        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader2012(
            args.datapath2012, log, split = False)

    elif args.datatype == 'mix':

        all_left_img_2015, all_right_img_2015, all_left_disp_2015, test_left_img_2015, test_right_img_2015, test_left_disp_2015 = ls.dataloader2015(
            args.datapath2015, log, split=False)
        all_left_img_2012, all_right_img_2012, all_left_disp_2012, test_left_img_2012, test_right_img_2012, test_left_disp_2012 = ls.dataloader2012(
            args.datapath2012, log, split=False)
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = \
            all_left_img_2015 + all_left_img_2012, all_right_img_2015 + all_right_img_2012, \
            all_left_disp_2015 + all_left_disp_2012, test_left_img_2015 + test_left_img_2012, \
            test_right_img_2015 + test_right_img_2012, test_left_disp_2015 + test_left_disp_2012
    else:

        AssertionError("please define the finetune dataset")

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))


    model = LWANet(args)


    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True

    start_full_time = time.time()



    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        if epoch % 100 == 0:
            savefilename = args.save_path + '/finetune_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)



        if epoch % 20 == 0:
            test(TestImgLoader, model, log)



    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))



def train(dataloader, model, optimizer, log, epoch=0):

    stages = 2
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = (disp_L > 0) & (disp_L < args.maxdisp)
        mask.detach_()

        pred, mono_loss = model(imgL, imgR)

        outputs = [torch.squeeze(output, 1) for output in pred]

        num_out = len(pred)
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]

        sum(loss).backward()

        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq == 0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(1)])
    log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_cspn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    total_inference_time = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):

        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():

            start_time = time.time()
            outputs, mono_loss = model(imgL, imgR)
            print(time.time() - start_time)
            total_inference_time += time.time() - start_time

            num_out = len(outputs)
            for x in range(num_out):

                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(num_out)])

        log.info('[{}/{}] {}'.format( batch_idx, length_loader, info_str))

    log.info("mean inference time:  %.3fs " % (total_inference_time / length_loader))
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(num_out)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):

    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()

    return err3.float() / mask.sum().float()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 600:
        lr = args.lr

    elif 600< epoch <= 1000:
        lr = args.lr*0.1

    else:
        lr = args.lr*0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()



