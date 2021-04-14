import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import numpy as np

class Jaccard(nn.Module):
    def __init__(self):
        super(Jaccard, self).__init__()

    def intersect(self, img, label):
        return torch.sum((img == label).to(dtype=torch.int64))

    def forward(self, img, label):
        intersection = self.intersect(img, label)
        union = img.shape[0]*img.shape[1] + label.shape[0]*label.shape[1] - intersection
        return intersection / union

    def to_tensor(self, label):
        label = torch.as_tensor(np.array(label), dtype=torch.int64)
        label = torch.transpose(label, 0,2)
        label = transforms.functional.rotate(label, 270)
        label = transforms.functional.hflip(label)
        return label[:][0]