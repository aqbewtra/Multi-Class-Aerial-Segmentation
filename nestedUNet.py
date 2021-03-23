import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
​
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
​
        self.basic_block = nn.Sequential(
            #conv1, bn, relu ---> block 1x
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
​
            #conv2, bn, relu ---> block 2
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.basic_block(x)
​
class NestedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters):
        super(NestedUNet, self).__init__()
​
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ### BACKBONE
​
        self.conv_0_0 = BasicBlock(in_channels=in_channels, out_channels=filters)
        self.conv_1_0 = BasicBlock(in_channels=filters, out_channels=filters*2)
        self.conv_2_0 = BasicBlock(in_channels=filters*2, out_channels=filters*4)
        self.conv_3_0 = BasicBlock(in_channels=filters*4, out_channels=filters*8)
        self.conv_4_0 = BasicBlock(in_channels=filters*8, out_channels=filters*16)
​
        ### WORK UP AND TO THE RIGHT
​
        self.conv_0_1 = BasicBlock(in_channels=filters+filters, out_channels=filters)
        self.conv_1_1 = BasicBlock(in_channels=filters*2+filters*2, out_channels=filters*2)
        self.conv_2_1 = BasicBlock(in_channels=filters*4+filters*4, out_channels=filters*4)
        self.conv_3_1 = BasicBlock(in_channels=filters*8+filters*8, out_channels=filters*8)
​
        self.conv_0_2 = BasicBlock(in_channels=filters+filters+filters, out_channels=filters)
        self.conv_1_2 = BasicBlock(in_channels=filters*2+filters*2+filters*2, out_channels=filters*2)
        self.conv_2_2 = BasicBlock(in_channels=filters*4+filters*4+filters*4, out_channels=filters*4)
​
        self.conv_0_3 = BasicBlock(in_channels=filters*4, out_channels=filters)
        self.conv_1_3 = BasicBlock(in_channels=(filters*2)*4, out_channels=filters*2)
​
        self.conv_0_4 = BasicBlock(in_channels=filters*5, out_channels=filters)
​
        self.out = nn.Conv2d(filters, out_channels, kernel_size=1)
​
        ### UPSAMPLES
​
        self.up1 = nn.ConvTranspose2d(in_channels=filters*2, out_channels=filters, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(in_channels=filters*4, out_channels=filters*2, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(in_channels=filters*8, out_channels=filters*4, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(in_channels=filters*16, out_channels=filters*8, kernel_size=2, stride=2)
​
    def forward(self, x):
        x0_0 = self.conv_0_0(x)
        x1_0 = self.conv_1_0(self.pool(x0_0))
        x2_0 = self.conv_2_0(self.pool(x1_0))
        x3_0 = self.conv_3_0(self.pool(x2_0))
        x = self.conv_4_0(self.pool(x3_0))
​
        x0_1 = self.conv_0_1(torch.cat([x0_0, self.up1(x1_0)], dim=1))
        x1_1 = self.conv_1_1(torch.cat([x1_0, self.up2(x2_0)], dim=1))
        x2_1 = self.conv_2_1(torch.cat([x2_0, self.up3(x3_0)], dim=1))
        x = self.conv_3_1(torch.cat([x3_0, self.up4(x)], dim=1))
​
        x0_2 = self.conv_0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], dim=1))
        x1_2 = self.conv_1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], dim=1))
        x = self.conv_2_2(torch.cat([x2_0, x2_1, self.up3(x)], dim=1))
​
        x0_3 = self.conv_0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], dim=1))
        x = self.conv_1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x)], dim=1))
​
        x = self.conv_0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x)], dim=1))
​
        return torch.sigmoid(self.out(x))
