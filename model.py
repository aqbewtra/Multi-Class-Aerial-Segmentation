import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        #conv1, bn, relu ---> block 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3
        )

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        #conv2, bn, relu ---> block 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3
        )

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(UNet, self).__init__()
        
        #block - double features, maxpool2d
        self.down1 = UNetBlock(in_channels=in_channels, out_channels=features)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block - double features, maxpool2d
        self.down2 = UNetBlock(in_channels=features, out_channels=features * 2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block - double features, maxpool2d
        self.down3 = UNetBlock(in_channels=features * 2, out_channels=features * 4)
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        #block - double features, maxpool2d
        self.down4 = UNetBlock(in_channels=features * 4, out_channels=features * 8)
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #bottleneck - double features without maxpool, before up conv
        # Base of the U structure
        self.bottleneck = UNetBlock(in_channels=features * 8, out_channels=features * 16)

        #block - half features, up-conv 2x2
        self.upConv4 = nn.ConvTranspose2d(in_channels=features * 16, out_channels=features * 8, kernel_size=2, stride=2)
        self.up4 = UNetBlock(in_channels=features * 16, out_channels=features * 8)

        #block - half features, up-conv 2x2
        self.upConv3 = nn.ConvTranspose2d(in_channels=features * 8, out_channels=features * 4, kernel_size=2, stride=2)
        self.up3 = UNetBlock(in_channels=features * 8, out_channels=features * 4)

        #block - half features, up-conv 2x2
        self.upConv2 = nn.ConvTranspose2d(in_channels=features * 4, out_channels=features * 2, kernel_size=2, stride=2)
        self.up2 = UNetBlock(in_channels=features * 4, out_channels=features * 2)

        #block - half features, up-conv 2x2
        self.upConv1 = nn.ConvTranspose2d(in_channels=features * 2, out_channels=features, kernel_size=2, stride=2)
        self.up1 = UNetBlock(in_channels=features * 2, out_channels=features)

        #conv 1x1 to output segmentation map
        self.outMap = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=3)

    def forward(self, x):
        #PAD ENCODED BLOCKS BEFORE CONCATTENATING TENSORS
        encode1 = self.down1(x)

        encode2 = self.maxPool1(encode1)
        encode2 = self.down2(encode2)

        encode3 = self.maxPool2(encode2)
        encode3 = self.down3(encode3)

        encode4 = self.maxPool3(encode3)
        encode4 = self.down4(encode4)

        bottleneck = self.maxPool4(encode4)
        bottleneck = self.bottleneck(bottleneck)

        decode = self.upConv4(bottleneck)
        #PAD
        decode = pad_to_match(decode, encode4)
        decode = torch.cat((decode, encode4), dim=1)
        decode = self.up4(decode)
        
        decode = self.upConv3(decode)
        decode = pad_to_match(decode, encode3)
        decode = torch.cat((decode, encode3), dim=1)
        decode = self.up3(decode)
        
        decode = self.upConv2(decode)
        decode = pad_to_match(decode, encode2)
        decode = torch.cat((decode, encode2), dim=1)
        decode = self.up2(decode)

        decode = self.upConv1(decode)
        decode = pad_to_match(decode, encode1)
        decode = torch.cat((decode, encode1), dim=1)
        decode = self.up1(decode)
        seg_map = self.outMap(decode)
        
        return torch.nn.Softmax(dim=1)(transforms.Resize((300,300))(seg_map))

def pad_to_match(small, big):
    diffX = big.size()[2] - small.size()[2]
    diffY = big.size()[3] - small.size()[3]

    return F.pad(small, [diffX // 2, diffX - (diffX // 2), diffY // 2, diffY - (diffY // 2)])