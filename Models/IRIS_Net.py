from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import  summary
from ptflops import get_model_complexity_info


class IRIS_Module(nn.Module):
    def __init__(self, in_channels, out_channels, connections = [1]):
        super().__init__()

        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels),
                                nn.BatchNorm2d(out_channels)
                                nn.ReLU(inplace=True)
                                )

        def forward(self, x):
            return module

class IRIS_Net(nn.Module):
    def __init__(self,
                 in_channels    =  3 , 
                 out_channels   = 1,
                 initial_depth  = 6, 
                 depth          = 14,
                 train_mode     = "segmentation"):
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.initial_depth  = initial_depth
        self.depth          = depth

        



    def forward(self, x):
        
        return out