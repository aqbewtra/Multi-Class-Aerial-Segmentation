import torch
import PIL.Image as Image

import torchvision.transforms as transforms

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from jaccard import Jaccard
import segm
from single_channel_util import formatting
from train import batch_size, num_workers
from dataset import SegmentationDataset
from util.pixel_accuracy import AccByClass, PixAcc
import matplotlib.pyplot as plt
from util import fix_labels

import numpy as np

model_path = 'model-7.pth'

dataset_root = '../data/dataset-sample/'
img_dir = dataset_root + 'image-chips/'
label_dir = dataset_root + 'label-chips/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = batch_size
num_workers = num_workers

def demo():
    #load model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    dataset = SegmentationDataset(img_dir, label_dir, scale=1)
    label = Image.open(dataset.path_dict[0][1]).convert('RGB')
    image = Image.open(dataset.path_dict[0][0]).convert('RGB')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            prediction = model(imgs)

            out_imgs = torch.argmax(prediction[0], dim=0)
            print(out_imgs)
            prediction.detach()
            break
    print(prediction[0].unique())
    jaccard = Jaccard()
    
    print("JACCARD:           ", jaccard(out_imgs, jaccard.to_tensor(label)/40))
    print("PIXEL ACCURACY:    ", PixAcc()(out_imgs, jaccard.to_tensor(label)/40), "\n")
    building, clutter, vegetation, water, ground, car = AccByClass()(out_imgs, jaccard.to_tensor(label)/40)
    print("0 BUILDING: {}\n1 CLUTTER: {}\n2 VEGETATION: {}\n3 WATER: {}\n4 GROUND: {}\n5 CAR: {}".format(building, clutter, vegetation, water, ground, car))
    print("\nJACCARD ignoring water and ground:", (building + clutter + vegetation + car) / 4)
    
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(out_imgs)
    axarr[1].imshow(label)
    axarr[2].imshow(image)
    plt.show()


if(__name__ == "__main__"):
    demo()