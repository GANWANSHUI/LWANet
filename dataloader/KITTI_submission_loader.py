import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader2015(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'


  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]


  left_test  = [filepath+left_fold+img for img in image]
  right_test = [filepath+right_fold+img for img in image]

  return left_test, right_test


def dataloader2012(filepath):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'


  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]


  left_test  = [filepath+left_fold+img for img in image]
  right_test = [filepath+right_fold+img for img in image]

  return left_test, right_test
