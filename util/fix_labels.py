import torch
from torchvision import transforms
from glob import glob
import os
from PIL import Image
import segm

labels_path = '../datadataset-sample/label-chips/'
def fix_labels(dir_path):
    print('go')
    label_paths = glob(os.path.join(dir_path, '*.png'))
    for i, path in enumerate(label_paths):
        print(i * 100 // len(label_paths), '%   : done')
        label = Image.open(path).convert("RGB")
        label = transforms.ToPILImage()((transforms.ToTensor()(label).unsqueeze_(0) * 40).squeeze_(0)).convert("RGB")
        label.save(path, "PNG")
        
'''
if __name__=="__main__":
    fix_labels(labels_path)
'''