#coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import argparse
np.set_printoptions(threshold=np.inf)
import torch.nn.functional as F
from PIL import Image
import utils.logger as logger
import time
from models.LWANet import *


parser = argparse.ArgumentParser(description='LWANet submission')

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[24, 3, 3])
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--with_cspn', type =bool, default= True, help='with cspn network or not')
parser.add_argument('--datapath', default='/data6/wsgan/SenceFlow/train/', help='datapath')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=8, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=8, help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='./results/kitti2015/benchmark', help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--print_freq', type=int, default=400, help='print frequence')

parser.add_argument('--model_types', type=str, default='original', help='model_types : LWANet_3D, mix, original')
parser.add_argument('--conv_3d_types1', type=str, default='separate_only', help='model_types :  3D, P3D ')
parser.add_argument('--conv_3d_types2', type=str, default='separate_only', help='model_types : 3D, P3D')
parser.add_argument('--cost_volume', type=str, default='Difference', help='cost_volume type :  "Concat" , "Difference" or "Distance_based" ')
parser.add_argument('--train', type =bool, default=True,  help='train or test ')


parser.add_argument('--datapath2015', default='/data6/wsgan/KITTI/KITTI2015/testing/', help='datapath')
parser.add_argument('--datapath2012', default='/data6/wsgan/KITTI/KITTI2012/testing/', help='datapath')
parser.add_argument('--datatype', default='2015', help='finetune dataset: 2012, 2015')

args = parser.parse_args()



if args.datatype == '2015':
   from dataloader import KITTI_submission_loader as DA

   test_left_img, test_right_img = DA.dataloader2015(args.datapath2015)

elif args.datatype == '2012':

   from dataloader import KITTI_submission_loader as DA
   test_left_img, test_right_img = DA.dataloader2012(args.datapath2012)

else:

    AssertionError("None found datatype")




log = logger.setup_logger(args.save_path + '/training.log')
for key, value in sorted(vars(args).items()):
    log.info(str(key) + ': ' + str(value))

if args.pretrained:
    if os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        log.info('=> loaded pretrained model {}'.format(args.pretrained))
    else:
        log.info('=> no pretrained model found at {}'.format(args.pretrained))
        log.info("=> Will start from scratch.")


else:
    log.info('Not Resume')



model = LWANet(args)
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()



def test(imgL,imgR):

    model.eval()
    if args.cuda:
       imgL = imgL.cuda()
       imgR = imgR.cuda()

    with torch.no_grad():
        disp, loss = model(imgL,imgR)

    disp = torch.squeeze(disp[-1])
    #print('disp size:', disp.shape)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp



def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])

    total_inference_time = 0

    for inx in range(len(test_left_img)):

        imgL_o = Image.open(test_left_img[inx]).convert('RGB')
        imgR_o = Image.open(test_right_img[inx]).convert('RGB')


        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)


        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)

        total_inference_time += time.time() - start_time

        if top_pad !=0 or right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        else:
            img = pred_disp

        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        print("inx:", inx)
        img.save(args.save_path  + test_left_img[inx].split('/')[-1])


    log.info("mean inference time:  %.3fs " % (total_inference_time/len(test_left_img)))

    log.info("finish {} images inference".format(len(test_left_img)))



if __name__ == '__main__':
    main()

