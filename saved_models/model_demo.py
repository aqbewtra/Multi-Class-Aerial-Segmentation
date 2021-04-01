import torch
import PIL.Image as Image

import torchvision.transforms as transforms

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import segm
from single_channel_util import formatting
from train import batch_size, num_workers
from dataset import SegmentationDataset

import matplotlib.pyplot as plt
from util import fix_labels

model_path = 'model-13.pth'

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
    dataset = SegmentationDataset(img_dir, label_dir, scale=1)
    image = Image.open(dataset.path_dict[0][1]).convert('RGB')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    #fig, axarr = plt.subplots(3,1)
    #run an image(s) through
    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            #segm.tensor_to_image(imgs[0])
            prediction = model(imgs)
            #convert out tensor to image
            ####CHANGE######
            # out_imgs = segm.tensor_to_image((prediction[0].unsqueeze_(0) * 40).squeeze_(0))
            #out_imgs = formatting.to_three_channel(prediction[0].squeeze())
            #out_imgs = fix_labels.


            out_imgs = torch.argmax(prediction[0], dim=0)
            print(out_imgs)
            prediction.detach()
            break
    print(prediction[0].unique())

    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(out_imgs)
    axarr[1].imshow(image)
    plt.show()


if(__name__ == "__main__"):
    demo()