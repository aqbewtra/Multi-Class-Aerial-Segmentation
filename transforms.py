import numpy as np

import PIL as pil
from PIL import Image, ImageFilter

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import segm
import random
from single_channel_util import formatting

# format image to tensor
def spatial_sampling(img_tensor, mode, **kwargs):
    if isinstance(img_tensor, (np.ndarray, pil.JpegImagePlugin.JpegImageFile, pil.Image.Image)):
        img_tensor = transforms.ToTensor()(img_tensor)
    img_tensor.unsqueeze_(0)
    img_tensor = F.interpolate(img_tensor, mode=mode, **kwargs)
    return img_tensor.squeeze(0)

def pool(img_tensor, kernel_size=3, stride=1, padding=1, mode='max', **kwargs):
    if isinstance(img_tensor, (np.ndarray, Image.Image)):
        img_tensor = transforms.ToTensor()(img_tensor)
    if mode.startswith('adaptive'):
        output_size = img_tensor.shape[1] if 'output_size' not in kwargs.keys() else kwargs['output_size']
    if mode == 'max':
        img_tensor = F.max_pool2d(img_tensor, kernel_size=kernel_size, stride=stride, padding=padding)
    elif mode == 'avg':
        img_tensor = F.avg_pool2d(img_tensor, kernel_size=kernel_size, stride=stride, padding=padding)
    elif mode == 'adaptive_max':
        img_tensor = F.adaptive_max_pool2d(img_tensor, output_size=output_size)
    elif mode == 'adaptive_avg':
        img_tensor = F.adaptive_avg_pool2d(img_tensor, output_size=output_size)
    elif mode == 'lp':
        ceil_mode = False if 'ceil_mode' not in kwargs.keys() else kwargs['ceil_mode']
        norm_type = 2.0 if 'norm_type' not in kwargs.keys() else kwargs['norm_type']
        img_tensor = F.lp_pool2d(img_tensor, kernel_size=kernel_size, stride=stride, \
            ceil_mode=ceil_mode, norm_type=norm_type)
    img_tensor.clamp_(min=0, max=1.)
    return img_tensor

class RandomAffine:
    def __init__(self, degrees=25, translate=(-0.1, 0.1), scale=(1.0, 0.15), shear=None, resample=False, fillcolor=0):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = np.random.normal(0, scale=degrees)
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)
        if scale_ranges is not None:
            mu, scale = scale_ranges
            scale = np.random.normal(loc=mu, scale=scale)
        else:
            scale = 1.0
        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0
        return angle, translations, scale, shear

    def __call__(self, img, label):
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return TF.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor), TF.affine(label, *ret, resample=self.resample, fillcolor=self.fillcolor)

def label_sampling(img_tensor, mode, **kwargs):
    if isinstance(img_tensor, (np.ndarray, pil.JpegImagePlugin.JpegImageFile, pil.Image.Image)):
        img_tensor = transforms.ToTensor()(img_tensor)

    ######CHANGE #######
    #img_tensor = segm.cvt_to_label(img_tensor)

    img_tensor = formatting.to_single_channel(img_tensor)
    
    img_tensor.unsqueeze_(0)
    #img_tensor = F.interpolate(img_tensor, mode=mode, **kwargs)
    return img_tensor.squeeze(0)

class SegmentationTransform:
    def __init__(self, scale, mode):
        self.scale = scale
        self.mode = mode
        self.sharpness = lambda: np.random.normal(loc=1.25, scale=0.2)
        self.gaussian = lambda: np.random.normal(loc=1.5, scale=0.5)
        self.gamma = lambda: np.random.normal(loc=1.25, scale=0.25)
        self.brightness = lambda: np.random.normal(loc=1.0, scale=0.25)

    def __call__(self, img, label):
        w, h = img.size
        img = img.filter(ImageFilter.GaussianBlur(radius=self.gaussian()))
        img = TF.adjust_gamma(img, gamma=self.gamma())
        img = TF.adjust_brightness(img, brightness_factor=self.brightness())
        if np.random.uniform() < 0.1:
            img = img.filter(ImageFilter.UnsharpMask(radius=self.sharpness()))
        #img, label = RandomAffine()(img, label)
        img = pool(img, mode='adaptive_avg', output_size=(int(self.scale * w), int(self.scale * h)))
        label = label_sampling(label, mode=self.mode, size=(int(self.scale * w), int(self.scale * h)))
        label = (label > 0.8).float()
        return img, label