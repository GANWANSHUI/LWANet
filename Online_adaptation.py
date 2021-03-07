import argparse
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import time
from dataloader import KITTIdatalist as ls
from dataloader import KITTILoader_0028_0071 as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import models
from models.LWANet import *


parser = argparse.ArgumentParser(description='Online adaptation')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1., 1., 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 3, 3])

parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--pretrained', type=str, default=None,
                    help='pretrained model path')
parser.add_argument('--model_types', type=str, default='original', help='model_types : LWANet_3D, mix, original')
parser.add_argument('--conv_3d_types1', type=str, default='separate_only', help='model_types :  normal, P3D, separate_only,  ONLY_2D ')
parser.add_argument('--conv_3d_types2', type=str, default='separate_only', help='model_types :  normal, P3D, separate_only,  ONLY_2D')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--with_cspn', default = True, action='store_true', help='with cspn network or not')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

parser.add_argument('--datapath0028', default='/data6/wsgan/KITTI/2011_09_30_drive_0028_sync/', help='datapath')
parser.add_argument('--datapath0071', default='/data6/wsgan/KITTI/2011_09_29_drive_0071/', help='datapath')

parser.add_argument('--datatype', default='0071',  help='datatype : 0028, 0071')
parser.add_argument('--adaptation_type', default='no_supervise',  help='adaptation_type : self_supervise, GT_supervise, no_supervise')
parser.add_argument('--save_path', type=str, default='results/video/0071/no_supervise/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--save_disparity', default = 'results/video/0071/no_supervise/disparity',  help='save disparity or not')


args = parser.parse_args()

# CUDA_VISIBLE_DEVICES=2 python finetune_adaptation_video.py  --with_cspn


def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    if args.datatype == "0028":
        train_left_img, train_right_img, train_left_disp = ls.dataloader_adaptation(
            args.datapath0028, args.datatype)


    elif args.datatype == "0071":
        train_left_img, train_right_img, train_left_disp = ls.dataloader_adaptation(
            args.datapath0071, args.datatype)



    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=False, num_workers=1, drop_last=False)



    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isdir(args.save_disparity):
        os.makedirs(args.save_disparity)


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


    cudnn.benchmark = True

    if args.adaptation_type == "self_supervise":
        model.train()
        loss_file = open(args.save_path + '/self_supervise' + '.txt', 'w')


    elif args.adaptation_type == "GT_supervise":
        model.train()
        loss_file = open(args.save_path + '/GT_supervise' + '.txt', 'w')


    elif args.adaptation_type == "no_supervise":

        loss_file = open(args.save_path + '/no_supervise' + '.txt', 'w')


    train(TrainImgLoader, model, optimizer, log, loss_file, args)



def train(dataloader, model, optimizer, log, loss_file, args):

    losses = [AverageMeter() for _ in range(2)]
    length_loader = len(dataloader)
    D1s = [AverageMeter() for _ in range(2)]

    start_full_time = time.time()
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        #print('train imgR size:', imgR.shape)

        optimizer.zero_grad()
        mask = disp_L > 0
        mask = mask*(disp_L<192)
        mask.detach_()

        single_update_time=time.time()

        #outputs = model(imgL, imgR)
        if args.adaptation_type == "no_supervise":
            model.eval()
            with torch.no_grad():
                pred, mono_loss = model(imgL, imgR)

            outputs = [torch.squeeze(output, 1) for output in pred]

            num_out = len(pred)
            loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                    for x in range(num_out)]


            num_out = len(pred)


        elif args.adaptation_type == "self_supervise":
            model.train()

            pred, mono_loss = model(imgL, imgR)
            outputs = [torch.squeeze(output, 1) for output in pred]
            num_out = len(pred)
            loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                    for x in range(num_out)]

            sum(mono_loss).backward()

            optimizer.step()

        elif args.adaptation_type == "GT_supervise":
            model.train()

            pred, mono_loss = model(imgL, imgR)

            outputs = [torch.squeeze(output, 1) for output in pred]

            num_out = len(pred)
            loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                    for x in range(num_out)]

            sum(loss).backward()
            optimizer.step()



        print('sigle_update_time: {:.4f} seconds'.format(time.time() - single_update_time))
        # image out and error estimation

        # three pixel error

        output = torch.squeeze(pred[1], 1)
        D1s[1].update(error_estimating(output, disp_L).item())
        print('output size:', output.shape)



        # save the adaptation disparity
        if args.save_disparity :

            plt.imshow(output.squeeze(0).cpu().detach().numpy())
            plt.axis('off')

            plt.gcf().set_size_inches(1216 / 100, 320 / 100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.savefig(args.save_path+'/disparity/{}.png'.format(batch_idx))

        # if args.save_disparity:
        #
        #     imgL = imgL.squeeze(0).permute(1,2,0)
        #     #print("imgL size:", imgL.shape)
        #     plt.imshow(imgL.cpu().detach().numpy())
        #     plt.axis('off')
        #
        #     plt.gcf().set_size_inches(1216 / 100, 320 / 100)
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #
        #     plt.savefig(args.save_path + '/disparity/{}.png'.format(batch_idx))
        #



        loss_file.write('{:.4f}\n'.format(D1s[1].val))

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())


        info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
        info_str = '\t'.join(info_str)

        log.info('Epoch{} [{}/{}] {}'.format(  1, batch_idx, length_loader, info_str))

    end_time = time.time()

    log.info('full training time = {:.2f} Hours, full train time = {:.4f} seconds'.format(
        (end_time - start_full_time) / 3600, end_time - start_full_time))

    # summary
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(num_out)])

    log.info('Average test 3-Pixel Error = ' + info_str)

    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(num_out)])
    log.info('Average train loss = ' + info_str)

    loss_file.close()



def error_estimating(disp, ground_truth, maxdisp=192):

    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()



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



