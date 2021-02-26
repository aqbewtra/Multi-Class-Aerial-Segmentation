
from PIL import Image
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import segm

sample_path = '../data/dataset-sample/label-chips/1d4fbe33f3_F1BE1D4184INSPIRE-000020.png'

def create_maps(six_channel):
    print(six_channel.size())
    map_list = []
    for i in range(six_channel.size[0]):
        map_list.append(six_channel[][][i])
    return map_list


def main():
    sample = Image.open(sample_path)
    #label to six_channel to tensor
    sample = segm.cvt_to_label(sample)
    #split tensor into tensor
    print(create_maps)
    #plot and show to test

if __name__ == "__main__":
    main()