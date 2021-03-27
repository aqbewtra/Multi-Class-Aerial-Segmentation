from glob import glob
import os
from PIL import Image

from torch.utils.data import Dataset

import matplotlib.pyplot as plt 
from transforms_ import SegmentationTransform



import torch
import numpy as np
from torchvision import transforms

from single_channel_util import formatting
import segm

#LABELS = ['BUILDING', 'CLUTTER', 'VEGETATION', 'WATER', 'GROUND', 'CAR'] indices * 40 are the label value now

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, label_dir, scale):
        if(not (0 < scale <= 1.0)):
            raise ValueError('Scale must be in (0, 1]; got {}'.format(scale))

        self.image_paths = glob(os.path.join(img_dir, '*.png'))
        self.label_paths = glob(os.path.join(label_dir, '*.png'))

        self.path_dict = dict()
        self.data_len = len(self.image_paths)

        for i, (i_path, m_path) in enumerate(zip(self.image_paths, self.label_paths)):
            self.path_dict.update({i: (i_path, m_path)})
        
        self.scale  = scale
        self.transform = SegmentationTransform(scale=scale)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        i_path, m_path = self.path_dict[index]
        img = Image.open(i_path).convert('RGB')
        label = Image.open(m_path).convert('RGB')
        img, label = self.transform(img, label)

        return img, label