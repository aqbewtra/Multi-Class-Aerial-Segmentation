import torch
from glob import glob
import os
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

def to_single_channel(label):
    return label[0]

def to_three_channel(tensor):
    tensor = torch.stack([tensor for i in range(3)], dim=0)

    return transforms.ToPILImage()(tensor).convert('RGB')