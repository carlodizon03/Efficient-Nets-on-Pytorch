import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary
from ptflops import get_model_complexity_info

class Dilation8(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()
        
        self.frontEnd1 = Dilation8._FrontEnd(in_channels = 3, out_channels = 64, count = 2, layer_num = 1, pool =True)
        self.frontEnd2 = Dilation8._FrontEnd(in_channels = 64, out_channels = 128, count = 2, layer_num = 2, pool =True)
        self.frontEnd3 = Dilation8._FrontEnd(in_channels = 128, out_channels = 256, count = 3, layer_num = 3, pool =True)
        self.frontEnd4 = Dilation8._FrontEnd(in_channels = 256, out_channels = 512, count = 3, layer_num = 4)
        self.frontEnd5 = Dilation8._FrontEnd(in_channels = 512, out_channels = 512, count = 3, layer_num = 5, dilation = 2)
        self.fc6       = Dilation8._fc()
        self.ctx       = Dilation8._context()
    def forward(self, x):
        x = self.frontEnd1(x)
        x = self.frontEnd2(x)
        x = self.frontEnd3(x)
        x = self.frontEnd4(x)
        x = self.frontEnd5(x)
        x = self.fc6(x)
        x = self.ctx(x)
        return x

    @staticmethod
    def _FrontEnd(in_channels, out_channels, count = 2, layer_num = 1, kernel_size = 3, pool = False, dilation = 1):
        layers = OrderedDict()
        layer_num = str(layer_num)
        pad = 0
        if dilation > 1:
            pad = 0
        for layer in range(1,count+1):
            if layer > 1:
                in_channels = out_channels
            layers['conv'+layer_num+'_'+str(layer)] = nn.Conv2d(
                                                                in_channels = in_channels,
                                                                out_channels = out_channels,
                                                                kernel_size = kernel_size,
                                                                padding = pad,
                                                                dilation = dilation
                                                                )
            layers['relu'+layer_num+'_'+str(layer)] = nn.ReLU(inplace=True)
        if pool:
            layers['pool'+layer_num+'_'] = nn.MaxPool2d(2,2)
        return nn.Sequential(layers)

    @staticmethod
    def _fc(in_channels = 512, out_channels = 11, dropout = 0.5):
        hidden_dim = in_channels*8
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc6",
                        nn.Conv2d(
                                    in_channels = in_channels,
                                    out_channels = hidden_dim, 
                                    kernel_size = 7,
                                    dilation = 4
                                )
                    ),
                    ("relu6", nn.ReLU(inplace = True)),
                    ("drop6", nn.Dropout(dropout, inplace = True)),
                    (
                        "fc7",
                        nn.Conv2d(
                                    in_channels = hidden_dim,
                                    out_channels = hidden_dim,
                                    kernel_size = 1,
                                )
                    ),
                    ("relu7", nn.ReLU(inplace = True)),
                    ("drop7", nn.Dropout(dropout, inplace = True)), 
                    (
                        "final",
                        nn.Conv2d(
                                    in_channels = hidden_dim,
                                    out_channels = out_channels,
                                    kernel_size = 1
                                )
                    )
                ]
        )
        )
    @staticmethod
    def _context(in_channels = 11, out_channels = 11):
        layers = OrderedDict()
        dils = [1,1,2,4,8,16,1,1]
        for idx, dil in enumerate(dils):
            pad = dil
            kernel_size = 3
            if idx == 7: 
                pad = 0
                kernel_size = 1
            layers['ctx_conv'+str(idx)+'_dil'+str(dil)] = nn.Conv2d(
                                                                        in_channels = in_channels,
                                                                        out_channels = out_channels,
                                                                        padding = pad,
                                                                        dilation = dil,
                                                                        kernel_size = kernel_size   
                                                                    )
            if idx < 7:
                layers['ctx_relu'+str(idx)] = nn.ReLU(inplace = True)
            else:
                layers['ctx_softmax'] = nn.Softmax()
        return nn.Sequential(layers)