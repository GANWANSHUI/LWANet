import argparse
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from dataloader import KITTILoader_One_cycle as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn

import models


from models.LWADNet import *


parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1., 1., 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/home/wsgan/KITTI_DATASET/KITTI2015/training/', help='datapath')
#parser.add_argument('--datapath', default='/home/um/GAN/Anynet/kitti2012/training/', help='datapath')

parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')


parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_cspn', action='store_true', help='with spn network or not')


parser.add_argument('--model_types', type=str, default='original', help='model_types : LWANet_3D, mix, original')
parser.add_argument('--conv_3d_types1', type=str, default='separate_only', help='model_types :  normal, P3D, separate_only,  ONLY_2D ')
parser.add_argument('--conv_3d_types2', type=str, default='separate_only', help='model_types :  normal, P3D, separate_only,  ONLY_2D')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')


parser.add_argument('--adaptation_type', default='GT_supervise',  help='adaptation_type : self_supervise, GT_supervise, no_supervise')

parser.add_argument('--pretrained', type=str, default='/home/wsgan/LWANet/results/pretrain/original_Difference/separate_only/checkpoint_49.tar',
                    help='pretrained model path')
parser.add_argument('--save_path', type=str, default='./results/finetune_One_cycle/GT_supervise/',
                    help='the path of saving checkpoints and log')



args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015_One_cycle as ls

elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls



# python One_cycle.py --with_cspn

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')
    #log1 = logger.setup_logger(args.save_path + '/self_adaptive_loss.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isdir(args.save_path+'/image'):
        os.makedirs(args.save_path+'/image')

    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))


    model = models.LWADNet.AnyNet(args)


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

    start_full_time = time.time()
    loss_file = open(args.save_path + '/self_supervise' + '.txt', 'w')

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))

        D1s= train(TrainImgLoader, model, optimizer, log, epoch)
        loss_file.write('{:.4f}\n'.format(D1s))


    loss_file.close()

    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):



    stages = 3 + args.with_cspn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    D1s = [AverageMeter() for _ in range(2)]


    model.train()

    #loss_file = open(args.save_path + '/self_adaptive_loss' + '.txt', 'w')

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        #print(' disp_L size:', disp_L)



        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()

        #outputs = model(imgL, imgR)
        pred, mono_loss = model(imgL, imgR)

        for x in range(len(pred)):
            output = torch.squeeze(pred[x], 1)
            D1s[x].update(error_estimating(output, disp_L).item())

        # loss_file.write('{:.4f}\n'.format(D1s[1].val))
        # loss_file.close()

       # print('len(outputs)', len(outputs))
        pred = [pred for pred in pred]
        num_out = len(pred)
        #print('num_out:', num_out)


        outputs = [torch.squeeze(output, 1) for output in pred]

        output_save = outputs[1].squeeze(0)
        #print('output_save:', output_save.shape)

        #io.imsave(args.save_path + '/epoch {}.png'.format(epoch), (output_save.cpu().data.numpy() ))

        plt.imshow(output_save.detach().cpu().numpy())
        plt.axis('off')

        #plt.savefig(args.save_path+'/image'+ '/epoch {} D1 {:.4f}.png'.format(epoch, D1s[1].val))
        plt.savefig(args.save_path + '/image' + '/epoch {} D1 {:.4f}.png'.format(epoch, D1s[1].val), bbox_inches = 'tight', dpi= 300, pad_inches = 0)


        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]

        #if args.adaptation_type == "no_supervise":

        #sum(mono_loss).backward()
        sum(loss).backward()
        #
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if 1:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))

            info_str = '\t'.join(
                ['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(num_out)])

            log.info('[{}/{}] {}'.format(
                batch_idx, length_loader, info_str))

        return  D1s[1].val


    # info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    # info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(2)])
    # log.info('Average train loss = ' + info_str)


def test(dataloader, model, log):

    stages = 3 + args.with_cspn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
       # print('test imgR size:', imgR.shape)

        # imgL = F.pad(imgL, [3, 3, 1, 0])
        # imgR = F.pad(imgR, [3, 3, 1, 0])
        # disp_L = F.pad(disp_L, [3, 3, 1, 0])
        #print('imgR  size:', imgR.shape)

        with torch.no_grad():
            outputs, mono_loss = model(imgL, imgR, train = 0)


            # for x in range(stages):
            if args.with_cspn:
                # if epoch >= args.start_epoch_for_spn:
                #     num_out = len(outputs)
                # else:
                #     num_out = len(outputs) - 1
                num_out = len(outputs)

            else:
                num_out = len(outputs)

            for x in range(num_out):
                output = torch.squeeze(outputs[x], 1)

                # print('output size:', output.shape)
                # print('disp_L size:', disp_L.shape)
                D1s[x].update(error_estimating(output, disp_L).item())


        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(num_out)])


        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))


    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(num_out)])

    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth


    # gt = gt[:, 0:368, 50:1200]
    # disp = disp[:, 0:368, 50:1200]
    # print('gt shape:', gt.shape)

    #mask = gt[:, 0:368, 50:1232]> 0

    mask = gt > 0
    mask = mask * (gt < maxdisp)

    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 1000:
        lr = args.lr
    elif epoch <= 1500:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
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


def post_process_disparity(disp):
    _, h, w = disp[0].shape
    #print('disp[0].shape:', disp[0].shape)  #  torch.Size([1, 368, 1232])

    l_disp = disp[0].cpu().numpy()
    #r_disp = np.fliplr(disp[1].cpu())
    r_disp = disp[1].cpu().numpy()

    #m_disp = 0.5 * (l_disp + r_disp)

    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    #r_mask =np.fliplr(l_mask)
    # return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    return l_mask * r_disp + (1.0 - l_mask ) * l_disp
    # benlaijiushi l_disp zhijiequdiao





if __name__ == '__main__':
    main()



