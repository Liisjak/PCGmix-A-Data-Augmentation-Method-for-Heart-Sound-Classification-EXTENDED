import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9_myrtle(nn.Module):
    def __init__(self, in_channels, num_classes, linear):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool2d = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.linear = nn.Linear(linear, num_classes) # for 128x128

    def forward(self, out, depth=None, pass_part=None):
        if pass_part=='first':
            if depth==0:
                return out
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.res1(out) + out
            if depth==1:
                return out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            if depth==2:
                return out
            out = self.pool2d(out)
            out = self.flat(out)
            if depth==3:
                return out
            out = self.linear(out)
            return out
        elif pass_part=='second':
            if depth<=0:
                out = self.conv1(out)
                out = self.conv2(out)
                out = self.res1(out) + out
            if depth<=1:
                out = self.conv3(out)
                out = self.conv4(out)
                out = self.res2(out) + out
            if depth<=2:
                out = self.pool2d(out)
                out = self.flat(out)
            if depth<=3:
                out = self.linear(out)
            return out
        else:
            #print('Input:', out.shape)
            out = self.conv1(out)
            #print('First conv:', out.shape)
            out = self.conv2(out)
            #print('Second conv:', out.shape)
            out = self.res1(out) + out
            #print('First res:', out.shape)
            out = self.conv3(out)
            #print('Third conv:', out.shape)
            out = self.conv4(out)
            #print('Forth conv:', out.shape)
            out = self.res2(out) + out
            #print('Second res:', out.shape)
            out = self.pool2d(out)
            #print('Pool before flat:', out.shape)
            out = self.flat(out)
            #print('Flat:', out.shape)
            out = self.linear(out)
            #print('Output:', out.shape)
            return out
        
def ResNet9(num_classes=2, linear=8192):
    return ResNet9_myrtle(in_channels = 1, num_classes=num_classes, linear=linear)
