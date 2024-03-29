from collections import OrderedDict
from numpy import double
import torch
import torch.nn as nn
import math
from torchsummary import  summary
from collections import defaultdict
from ptflops import get_model_complexity_info
from logger import Logger
from torchstat import stat
import pandas as pd
import matplotlib.pyplot as plt
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1):
        super().__init__()
        self.add_module('conv2d',nn.Conv2d(in_channels,out_channels,kernel_size,stride))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU() )
    def forward(self, input):
        return super().forward(input)
class fibModule(nn.Module):
    '''
        Fib module is going to use the connection sequence of fibonacci sequence.

        Connections:

        Where Kth layer will be a convolution operation of concatenated outputs of K-1 and K-2 layers,
        if Kth layer is greater than 2 and K is equal to depth of module.

        Growth rate:
            WIP

            Requirements: 
                    1) Memory footprint efficient
                    2) Sparse series of convolution/operations


    '''
    def __init__(self, in_channels = 3, num_blocks = 5, block_depth = 5):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.encoder = self.build(in_channels = self.in_channels, num_blocks = self.num_blocks, block_depth = self.block_depth)
        
    def fibonacci(self,depth):

        f = []
        for i in range(1,depth+1):
            num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
            den = 2**i * (math.sqrt(5))
            f.append(int(num/den))
        return(f)

    def naive_block_channels_variation(self, blocks, in_channels = 3,  depth = 5, ratio = 0.618):
        channel_list =[in_channels]
        ratio_list = [ratio]
        print(blocks)
        for block_idx, block in enumerate(blocks):
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list.append(val)
                ratio_ = 3.414 * ratio_ * (1 - ratio_)
                depth_ -= 1
                ratio_list.append(ratio_)
        # print(len(channel_list))
        print(channel_list)
        print(ratio_list)

        # plt.plot(channel_list)
        plt.plot(ratio_list)
        plt.show()
        return channel_list   

    

    def build(self, in_channels = 3, num_blocks = 5, block_depth = 5):
        initial_ratio = 0.618
        blocks_channel_list= self.naive_block_channels_variation(blocks = self.fibonacci(num_blocks),  in_channels = in_channels, depth = block_depth)
        encoder = nn.ModuleList()
        
        for block in range(num_blocks):
            
            in_channels = blocks_channel_list[block*block_depth]
            out_channels = blocks_channel_list[block*block_depth+1]
            #Conv2d to match the shape for concatenation
            encoder.append(ConvLayer(in_channels, 
                                    in_channels, 
                                    kernel_size = 3,
                                    stride = 1))
            #start of block conv
            encoder.append(ConvLayer(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride = 1))
            # print(block, 0, in_channels,out_channels)
            for layer in range(1,block_depth):
                idx =  block*block_depth+layer
                in_channels = blocks_channel_list[idx] + blocks_channel_list[idx-1]
                out_channels = blocks_channel_list[idx+1]
                encoder.append(ConvLayer(in_channels = blocks_channel_list[idx],
                                              out_channels = blocks_channel_list[idx],
                                              kernel_size=3))
                encoder.append(ConvLayer(in_channels = in_channels,
                                              out_channels = out_channels,
                                              kernel_size=3))
                # print(block, layer, blocks_channel_list[idx],blocks_channel_list[idx])
                # print(block, layer, in_channels,out_channels)
                if idx +1 == block_depth * num_blocks:
                    break
        # for idx, layer in enumerate(encoder):
        #     print(idx, layer)
        # print("--------------------------------------------")
        return encoder
        
    def forward(self, inputs):
        x = inputs 
        for block in range(self.num_blocks):
            # print('cat idx:{0} \t out_idx: {1}'.format(block*self.block_depth*2, block*self.block_depth*2+1))
            cat_out = self.encoder[block*self.block_depth*2](x)
            # print(self.encoder[block*self.block_depth*2])
            out = self.encoder[block*self.block_depth*2+1](x)
            # print(self.encoder[block*self.block_depth*2+1])
            # print('block: {0} \t layer: {1} \t x: {2}  \t cat_out: {3} \t out: {4}'.format(block, 0, x.shape,  cat_out.shape, out.shape))
            for layer in range(1,self.block_depth):
                # print(out.shape)
                

                in2 = torch.cat((out,cat_out),1)
                # print()
                # print("--------------------Concatenate-----------------------")
                # print()
                x = out
                # print('cat idx:{0} \t out_idx: {1}'.format(block*self.block_depth*2+(layer*2), block*self.block_depth*2+(layer*2)+1))
                cat_out = self.encoder[block*self.block_depth*2+(layer*2)](x)
                # print(self.encoder[block*self.block_depth*2+(layer*2)])

                out  = self.encoder[block*self.block_depth*2+(layer*2)+1](in2)
                # print(self.encoder[block*self.block_depth*2+(layer*2)+1])

                # print('block: {0} \t layer: {1} \t x: {2} \t in2:{3} \t cat_out: {4} \t out: {5}'.format(block, layer, x.shape, in2.shape, cat_out.shape, out.shape))
                # print(layer)
                if layer == self.block_depth-1:
                    x = out
                    # print("later: 4 \t x:{0}".format(x.shape))
                    # print("----------------------End of Block----------------------")
        return out

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 5, block_depth = 5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.conv1 = ConvLayer(3,16,3,2)
        self.conv2 = ConvLayer(16,32,3,2)
        self.encoder = fibModule(in_channels = 32, num_blocks = self.num_blocks, block_depth = self.block_depth)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(49, self.out_channels)
    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        outputs = self.encoder(inputs)
        outputs = self.avgpool(outputs)
        # outputs = outputs.view(outputs.size(0), -1)
        # outputs = self.classifier(outputs)
        return outputs
# output shape = [(W−K+2P)/S]+1
"""Load Cuda """
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
""""""""""""""""""
# log = Logger(path = 'logs/fibnet')
f = FibNet(in_channels=3,out_channels=1000, num_blocks=5, block_depth=5)
# f.cuda()
# x = torch.rand(3,224,224)
# x = x.type(torch.cuda.FloatTensor)
# y = f(x)
# print(y)
# summary(f,(3,224,224))
# log.model_graph(f, x)
# macs, params = get_model_complexity_info(f, (3, 224, 224), as_strings=True,
#                                         print_per_layer_stat=False, verbose=False)
# print()
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# df = stat(f,(3,224,224))

# in_shape = [val for val in df.loc[:,'input shape'].values]
# out_shape = [val for val in df.loc[:,'output shape'].values]

# mems_read = [val/1000000 for val in df.loc[:,'MemRead(B)'].values]
# mems_write = [val/1000000 for val in df.loc[:,'MemWrite(B)'].values]
# mems_rw = [val/1000000 for val in df.loc[:,'MemR+W(B)'].values]

# names = [val.replace('encoder.','') for val in df.loc[:,'module name'].values]
# for idx, name in enumerate(names):
#   print(name.replace('encoder.',''))


# print('-------------input shapes------------------')
# for ins in in_shape:
#   print(ins)
# print('-------------output shapes------------------')
# for out in out_shape:
#   print(out)
# print('-------------Memory Read------------------')
# for r in mems_read:
#   print(r)
# print('-------------Memory write------------------')
# for w in mems_write:
#   print(w)
# print('-------------Memory readwrite------------------')

# for rw in mems_rw:
#   print(rw)