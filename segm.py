import torch
from glob import glob
import os
from PIL import Image
from torchvision import transforms

def cvt_to_label(y):
    """
    # read in label 
        y = Image.open('path')
    # convert from rgb to tensor
        y = TF.to_tensor(y)
    # convert to correct format
        label = cvt_to_label(y)
    """
    labels = torch.unique(find_complete_label_tensor(sample_label_path, sample_label_path_2))

    
    
    return torch.stack([((y == labels[i]).sum(dim=0) > 0) for i in range(labels.size(0))], dim=0)

label_dir = 'data/dataset-sample/label-chips/'

sample_label_path = 'data/dataset-sample/label-chips/1d4fbe33f3_F1BE1D4184INSPIRE-000020.png'
sample_label_path_2 = 'data/dataset-sample/label-chips/ec09336a6f_06BA0AF311OPENPIPELINE-000032.png'
#BE CAREFUL IF ORDER FOF IMAGES CHANGES
def find_complete_label_tensor(img1, img2):
    '''
    label_paths = glob(os.path.join(label_dir, '*.png'))
    sample_path = None
    label_categories = None
    for i, path in enumerate(label_paths):
        y = Image.open(path).convert('RGB')
        y = transforms.ToTensor()(y)
        y = torch.unique(y)
        
        
        if(y.size() == torch.Size([5])):
            print(i, y)
        
            sample_path = path
            label_categories = y
            break
        
    return sample_path, label_categories

    '''

    y = Image.open(img1).convert('RGB')
    y = transforms.ToTensor()(y)
    z = Image.open(img2).convert('RGB')
    z = transforms.ToTensor()(z)
    y = torch.stack([y,z], dim=0)
    y = torch.unique(y)
    return y


if __name__ == "__main__":
    print(find_complete_label_tensor(sample_label_path, sample_label_path_2))