import torch
from glob import glob
import os
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

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
    try:
        y = Image.open(img1).convert('RGB')
        z = Image.open(img2).convert('RGB')

    except FileNotFoundError:
        y = Image.open('../'+img1).convert('RGB')
        z = Image.open('../'+img2).convert('RGB')
        
    
    y = transforms.ToTensor()(y)
    
    z = transforms.ToTensor()(z)
    y = torch.stack([y,z], dim=0)
    y = torch.unique(y)
    return y



def tensor_to_image(tens):
    rgb_values = find_complete_label_tensor(sample_label_path, sample_label_path_2)
    interim_tens = torch.zeros([tens.shape[1], tens.shape[2]], dtype=torch.float32)
    
    for i, color in enumerate(rgb_values):
        interim_tens += (tens[i] * color)
    
    interim_tens = torch.stack([interim_tens for i in range(3)], dim=0)

    return transforms.ToPILImage()(interim_tens).convert('RGB')
        


if __name__ == "__main__":
    print('HERE', find_complete_label_tensor(sample_label_path, sample_label_path_2))
    y = Image.open(sample_label_path).convert('RGB')
    y = transforms.ToTensor()(y)
    y = cvt_to_label(y)
    
    y = tensor_to_image(y)

    plt.imshow(y)
    plt.show()



