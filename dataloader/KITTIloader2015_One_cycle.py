import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, log):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'
  #disp_R = 'disp_occ_1/'
  #
  # left_fold = 'image_02/data/'
  # right_fold = 'image_03/data/'
  # disp_L = 'data_depth_annotated/2011_09_30_drive_0028_sync/proj_depth/groundtruth/image_02/'
  #disp_R = 'disp_occ_1/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  #image = [img for img in os.listdir(filepath + left_fold) if img.find('000000_10')]
  #print('image 0:', len(image))

  all_index = np.arange(200)
  #np.random.seed(2)
  #np.random.shuffle(all_index)
  #print('all_index:', all_index)
  vallist = all_index[:40]

  log.info(vallist)
  val = ['{:06d}_10.png'.format(x) for x in vallist]
  #train = [x for x in image if x not in val]
  train = [x for x in image if x == '000128_10.png']
  print('train :', train[0])





  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
  #return left_train, right_train, disp_train_L, disp_train_R, left_val, right_val, disp_val_L, disp_val_R