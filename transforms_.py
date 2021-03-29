import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import random
import torchvision.transforms as transforms
import torch

class SegmentationTransform:
    def __init__(self, scale):
        self.scale = scale
        self.sharpness = lambda: np.random.normal(loc=1.25, scale=0.2)
        self.gaussian = lambda: np.random.normal(loc=1.5, scale=0.5)
        self.gamma = lambda: np.random.normal(loc=1.25, scale=0.25)
        self.brightness = lambda: np.random.normal(loc=1.0, scale=0.25)

    def __call__(self, img, label):
        # w, h = img.size
        img = img.filter(ImageFilter.GaussianBlur(radius=self.gaussian()))
        img = TF.adjust_gamma(img, gamma=self.gamma())
        img = TF.adjust_brightness(img, brightness_factor=self.brightness())
        if np.random.uniform() < 0.1:
            img = img.filter(ImageFilter.UnsharpMask(radius=self.sharpness()))
        img, label = RandomAffine()(img, label)
        img = transforms.Resize((256,256))(img)
        label = transforms.Resize((256, 256))(label)
        img, label = to_tensor(img, label)
        return img, label

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



def to_tensor(img, label):
    img = transforms.ToTensor()(img)
    label = torch.as_tensor(np.array(label), dtype=torch.int64)
    label = torch.transpose(label, 0,2)
    label = transforms.functional.rotate(label, 270)
    label = transforms.functional.hflip(label)
    return img, label[:][0]