from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import  summary
from ptflops import get_model_complexity_info


class Projection(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32, name = "layer"):
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "depthwise_1",
                        nn.Conv2d(
                            in_channels = in_channels,out_channels = in_channels, 
                            kernel_size = 3, stride = 1, padding= 1, groups = in_channels,
                            bias = False  
                        )
                    ),
                    (
                        name + "pointwise_1",
                        nn.Conv2d(
                            in_channels = in_channels, out_channels = out_channels,
                            kernel_size = 1)
                        
                    ),

                    (
                        name + "depthwise_2",
                        nn.Conv2d(
                            in_channels = out_channels,out_channels = out_channels, 
                            kernel_size = 3, stride = 1, padding= 1, groups = out_channels,
                            bias = False  
                        )
                    ),
                    (
                        name + "pointwise_2",
                        nn.Conv2d(
                            in_channels = out_channels, out_channels = out_channels,
                            kernel_size = 1
                            )
                    ),

                    (name + "norm1", nn.BatchNorm2d(num_features = out_channels)),
                    (name + "relu1", nn.ReLU6(inplace = True)),

                    (
                        name + "conv",
                        nn.Conv2d(
                            in_channels = out_channels, out_channels = out_channels, 
                            kernel_size = 3, padding = 1, bias = False
                        )
                    ),

                    (name + "norm2", nn.BatchNorm2d(num_features = out_channels)),
                    (name + "relu2", nn.ReLU6(inplace = True))

                ]
            )
        )
    def forward(self,x):
        return self.block(x)
        

# """Load Cuda """
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True
# """"""""""""""""""
# model = Projection()
# model.to(device)
# summary(model, (3,224,224))