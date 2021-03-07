import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader2012(filepath, log, split=False):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_occ/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  random.shuffle(image)


  if not split:

    np.random.seed(2)
    random.shuffle(image)
    train = image[:]
    val = image[160:]

  else:

    train = image[:160]
    val   = image[160:]



  log.info(val)

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train = [filepath+disp_noc+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val = [filepath+disp_noc+img for img in val]

  return left_train, right_train, disp_train, left_val, right_val, disp_val



def dataloader2015(filepath, log, split = False):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  all_index = np.arange(200)
  np.random.seed(2)
  np.random.shuffle(all_index)
  #print('all_index:', all_index)
  vallist = all_index[:40]

  log.info(vallist)
  val = ['{:06d}_10.png'.format(x) for x in vallist]

  if split:
    train = [x for x in image if x not in val]
  # train = [x for x in image if x not in val]

  else:
    train = [x for x in image]



  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]
  #disp_val_R = [filepath+disp_R+img for img in val]


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L




def dataloader_adaptation(filepath, datatype):

  # 0028
  left_fold = 'raw_image/image_02/data/'
  right_fold = 'raw_image/image_03/data/'   #  w, h: 1226 370
  disp_L = 'disparity/image_02/'

  path_list = os.listdir(filepath + left_fold)
  path_list.sort(key=lambda x: int(x.split('.')[0]))
  image = [img for img in path_list]


  #0028
  if datatype == "0028":
    image = image[5:2005]


  elif datatype == "0071":
    # 0071
    image = image[5:-6]


  train = [x for x in image]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]

  return left_train, right_train,