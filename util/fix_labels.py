import torch
from torchvision import transforms
from glob import glob
import os
import sys
from PIL import Image
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import segm
import matplotlib.pyplot as plt

labels_path = '../data/fix_labels/'

def fix_labels(dir_path):
    print('go')
    label_paths = glob(os.path.join(dir_path, '*.png'))
    for i, path in enumerate(label_paths):
        print(i * 100 // len(label_paths), '%   : done')
        label = Image.open(path).convert("RGB")
        label = transforms.ToPILImage()((transforms.ToTensor()(label).unsqueeze_(0) * 40).squeeze_(0)).convert("RGB")
        label.save(path, "PNG")


def fix_labels2(label):
    label = segm.cvt_to_label(label).to(dtype=torch.float32)
    for i in range(len(label)):
        label[i] = label[i] * i
    interim_tens = torch.zeros([label.shape[1], label.shape[2]], dtype=torch.float32)
    for i in range(len(label)):
        interim_tens += (label[i])
    return interim_tens
    
    #return label

'''
if __name__=="__main__":
    #fix_labels(labels_path)
'''