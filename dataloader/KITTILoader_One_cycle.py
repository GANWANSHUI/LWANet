import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)



        w, h = left_img.size

        left_img = left_img.crop((w - 1216, h - 320, w, h))
        right_img = right_img.crop((w - 1216, h - 320, w, h))



        dataL = dataL.crop((w-1216, h-320, w, h))

        dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        processed = preprocess.get_transform(augment=False)
        left_img       = processed(left_img)
        right_img      = processed(right_img)



        return left_img, right_img, dataL






    def __len__(self):
        return len(self.left)



