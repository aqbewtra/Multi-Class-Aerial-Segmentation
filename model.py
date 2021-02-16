import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        #conv1, bn, relu ---> block 1
        #UNPADDED????
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
'''
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            UNetBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNetUp, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2),
        
        self.conv = UNetBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
'''

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(UNet, self).__init__()
        
        #block - double features, maxpool2d
        self.down1 = UNetBlock(
            in_channels=in_channels,
            out_channels=features,
        )
        self.maxPool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        #block - double features, maxpool2d
        self.down2 = UNetBlock(
            in_channels=features,
            out_channels=features * 2
        )
        self.maxPool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        #block - double features, maxpool2d
        self.down3 = UNetBlock(
            in_channels=features * 2,
            out_channels=features * 4
        )
        self.maxPool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        #block - double features, maxpool2d
        self.down4 = UNetBlock(
            in_channels=features * 4,
            out_channels=features * 8
        )
        self.maxPool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        #bottleneck - double features without maxpool, before up conv
        # Base of the U structure
        self.bottleneck = UNetBlock(
            in_channels=features * 8,
            out_channels=features * 16
        )

        #block - half features, up-conv 2x2
        self.upConv4 = nn.ConvTranspose2d(
            in_channels=features * 16,
            out_channels=features * 8,
            kernel_size=2,
            stride=2
        )
        self.up4 = UNetBlock(
            in_channels=features * 16,
            out_channels=features * 8
        )

        #block - half features, up-conv 2x2
        self.upConv3 = nn.ConvTranspose2d(
            in_channels=features * 8,
            out_channels=features * 4,
            kernel_size=2,
            stride=2
        )
        self.up3 = UNetBlock(
            in_channels=features * 8,
            out_channels=features * 4
        )

        #block - half features, up-conv 2x2
        self.upConv2 = nn.ConvTranspose2d(
            in_channels=features * 4,
            out_channels=features * 2,
            kernel_size=2,
            stride=2
        )
        self.up2 = UNetBlock(
            in_channels=features * 4,
            out_channels=features * 2 
        )

        #block - half features, up-conv 2x2
        self.upConv1 = nn.ConvTranspose2d(
            in_channels=features * 2,
            out_channels=features,
            kernel_size=2,
            stride=2
        )
        self.up1 = UNetBlock(
            in_channels=features * 2,
            out_channels=features
        )

        #conv 1x1 to output segmentation map
        self.outMap = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=3
        )

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
        seg_map = torch.sigmoid(self.outMap(decode))
        
        return F.pad(seg_map, [5,5,5,5])

        #return torch.sigmoid()

def pad_to_match(small, big):
    diffX = big.size()[2] - small.size()[2]
    diffY = big.size()[3] - small.size()[3]

    small = F.pad(small, [diffX // 2, diffX - (diffX // 2), diffY // 2, diffY - (diffY // 2)])

    return small


    

    

'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = UNetBlock(n_channels, 64)
        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = UNetDown(512, 1024 // factor)
        self.up1 = UNetUp(1024, 512 // factor, bilinear)
        self.up2 = UNetUp(512, 256 // factor, bilinear)
        self.up3 = UNetUp(256, 128 // factor, bilinear)
        self.up4 = UNetUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)





    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
'''