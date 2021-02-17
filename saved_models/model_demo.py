import torch
import PIL.Image as Image

import torchvision.transforms as transforms

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import segm
from train import batch_size, num_workers
from dataset import SegmentationDataset

import matplotlib.pyplot as plt

model_path = 'model-0(1).pth'

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

    #dataset
    dataset = SegmentationDataset(img_dir, label_dir, scale=1, mode='nearest')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    fig = plt.figure()
    #run an image(s) through
    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            #segm.tensor_to_image(imgs[0])
            #plt.imshow(segm.tensor_to_image(imgs[0]))
            
            prediction = model(imgs)
            
            #convert out tensor to image
            out_imgs = segm.tensor_to_image(prediction[0])
            prediction.detach()
            break
    print(prediction[0][1].unique())
    #plt show image
    fig.add_subplot()
    plt.imshow(out_imgs)
    plt.show()  


if(__name__ == "__main__"):
    demo()