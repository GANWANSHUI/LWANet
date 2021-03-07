import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from torch.autograd import Variable
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger
from utils.flops_hook import profile
from models.LWANet import *


parser = argparse.ArgumentParser(description='LWANet with Sceneflow dataset')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 3, 3])
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--with_cspn', type =bool, default= True, help='with cspn network or not')
parser.add_argument('--datapath', default='/data6/wsgan/SenceFlow/train/', help='datapath')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=16, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=8, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='./results/sceneflow/', help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--print_freq', type=int, default=400, help='print frequence')

parser.add_argument('--model_types', type=str, default='LWANet', help='model_types : 3D, P3D')
parser.add_argument('--conv_3d_types1', type=str, default='P3D', help='model_types :  3D, P3D ')
parser.add_argument('--conv_3d_types2', type=str, default='P3D', help='model_types : 3D, P3D')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--train', type =bool, default=True,  help='train or test ')


args = parser.parse_args()

#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py


def main():
    global args

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)
    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)
    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))


    model = LWANet(args)


    # FLOPs, params = count_flops(model.cuda())
    # log.info('Number of model parameters: {}'.format(params))
    # log.info('Number of model FLOPs: {}'.format(FLOPs))


    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()

    if args.train:
        for epoch in range(args.start_epoch, args.epochs):
            log.info('This is {}-th epoch'.format(epoch))

            train(TrainImgLoader, model, optimizer, log, epoch)

            savefilename = args.save_path + '/checkpoint_' + str(epoch) + '.tar'

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, savefilename)

            if not epoch % 10:
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
        mask = (disp_L < args.maxdisp) & (disp_L > 0)
        if mask.float().sum() == 0:
            continue

        mask.detach_()

        outputs, self_supervised_loss = model(imgL, imgR)
        stages = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]

        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]

        sum(loss).backward()
        optimizer.step()

        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])

        if batch_idx % args.print_freq ==0:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)



def test(dataloader, model, log):

    stages = 2
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    inference_time = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp
        with torch.no_grad():

            time_start = time.perf_counter()


            outputs, monoloss = model(imgL, imgR)

            single_inference_time = time.perf_counter() - time_start

            inference_time += single_inference_time


            stages = len(outputs)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                output = output[:, 4:, :]
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        if batch_idx % args.print_freq == 0:
            info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])

            log.info('[{}/{}] {}'.format(
                batch_idx, length_loader, info_str))

    log.info(('=> Mean inference time for %d images: %.3fs' % (
        length_loader, inference_time / length_loader)))

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
        lr = args.lr

    elif 20 <epoch <= 30:
        lr = args.lr * 0.5

    elif 30< epoch <= 40:
        lr = args.lr * 0.25

    else:
        lr = args.lr * 0.125


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def count_flops(model):

    input = Variable(torch.randn(1, 3, 544, 960)).cuda()

    FLOPs, params = profile(model, inputs = (input,input))

    return FLOPs, params


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



# 快捷指令

# https://www.cnblogs.com/kaiye/p/6275207.html

# tmux new -s name  创建session
# tmux a -t  name 进入session
# tmux ls 查看session
# tmux kill-session -t name 杀掉session 进程
# ctrl + B 松开 D 退出当前session的全部窗口
# ctrl + B 松开+ C 基于当前session新建窗口


# exit 退出当前bash的窗口
# ctrl +B 松开 + x 删除当前bash


# ctrl + B  松开 -   水平分屏
# ctrl + B 松开 shift + - 垂直分屏

# tmux kill-server # 删除所有的会话

