import torch
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import segm
from single_channel_util import formatting
from train import batch_size, num_workers
from dataset import SegmentationDataset
from transforms import SegmentationTransform
from glob import glob

import matplotlib.pyplot as plt

model_path = 'model-7.pth'

img_dir = '../data/images-selected/'

label_dir = '../data/images-selected/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
num_workers = num_workers

class AirSimDataset(Dataset):
    def __init__(self, img_dir, label_dir, scale, mode):
        if(not (0 < scale <= 1.0)):
            raise ValueError('Scale must be in (0, 1]; got {}'.format(scale))
        # get image and label paths
        self.image_paths = glob(os.path.join(img_dir, '*.png'))
        self.label_paths = glob(os.path.join(label_dir, '*.png'))
        print(self.image_paths)

        # create path dict
        self.path_dict = dict()
        self.data_len = len(self.image_paths)


        # for look: enumerate the paths list, and use integer keys for tuples (image_path, label path)
        for i, (i_path, m_path) in enumerate(zip(self.image_paths, self.label_paths)):
            self.path_dict.update({i: (i_path, m_path)})
        
        self.scale  = scale
        self.transform = SegmentationTransform(scale=scale, mode=mode)

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        i_path, m_path = self.path_dict[index]
        img = Image.open(i_path).convert('RGB')
        label = Image.open(m_path).convert('RGB')
        img, label = self.transform(img, label)
        img = transforms.Compose([transforms.Resize((300,300))])(img)

        return img, label

def demo():
    #load model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    index = 0
    #dataset
    dataset = AirSimDataset(img_dir, label_dir, scale=1, mode='nearest')
    image = Image.open(dataset.path_dict[index][1]).convert('RGB')
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, \
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    with torch.set_grad_enabled(False):
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = map(lambda x: x.to(device, dtype=torch.float32), (imgs, labels))
            prediction = model(imgs)

            out_imgs = torch.argmax(prediction[index], dim=0)
            print(out_imgs)
            prediction.detach()
            if batch_idx == 1:
                break
    print(prediction[index].unique())

    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(out_imgs)
    axarr[1].imshow(image)
    plt.show()  


if(__name__ == "__main__"):
    demo()