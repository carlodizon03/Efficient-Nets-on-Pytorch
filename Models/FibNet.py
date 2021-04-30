from collections import OrderedDict
import torch
import torch.nn as nn
import math
from torchsummary import  summary
from collections import defaultdict
from ptflops import get_model_complexity_info

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
        for block_idx, block in enumerate(blocks):
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list.append(val)
                ratio_ = 3.414 * ratio_ * (1 - ratio_)
                depth_ -= 1
        return channel_list   

    def convLayer(self, in_channels, out_channels, kernel_size = 3, stride = 1):
        return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU())    
    def build(self, in_channels = 3, num_blocks = 5, block_depth = 5):
        initial_ratio = 0.618
        blocks_channel_list= self.naive_block_channels_variation(self.fibonacci(num_blocks),  in_channels, block_depth)
        encoder = nn.ModuleList()
        
        for block in range(num_blocks):
            
            in_channels = blocks_channel_list[block*block_depth]
            out_channels = blocks_channel_list[block*block_depth+1]
            #Conv2d to match the shape for concatenation
            encoder.append(self.convLayer(in_channels, 
                                    in_channels, 
                                    kernel_size = 3,
                                    stride = 1))
            #start of block conv
            encoder.append(self.convLayer(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride = 1))
            # print(block, 0, in_channels,out_channels)
            for layer in range(1,block_depth):
                idx =  block*block_depth+layer
                in_channels = blocks_channel_list[idx] + blocks_channel_list[idx-1]
                out_channels = blocks_channel_list[idx+1]
                encoder.append(self.convLayer(in_channels = blocks_channel_list[idx],
                                              out_channels = blocks_channel_list[idx],
                                              kernel_size=3))
                encoder.append(self.convLayer(in_channels = in_channels,
                                              out_channels = out_channels,
                                              kernel_size=3))
                # print(block, layer, blocks_channel_list[idx],blocks_channel_list[idx])
                # print(block, layer, in_channels,out_channels)
                if idx +1 == block_depth * num_blocks:
                    break
        for idx, layer in enumerate(encoder):
            print(idx, layer)
        return encoder

    def forward(self, inputs):
        x = inputs 
        for block in range(self.num_blocks):
            # print('cat idx:{0} \t out_idx: {1}'.format(block*self.block_depth*2, block*self.block_depth*2+1))
            cat_out = self.encoder[block*self.block_depth*2](x)
            out = self.encoder[block*self.block_depth*2+1](x)
            # print('block: {0} \t layer: {1} \t x: {2}  \t cat_out: {3} \t out: {4}'.format(block, 0, x.shape,  cat_out.shape, out.shape))
            for layer in range(1,self.block_depth):
                in2 = torch.cat((out,cat_out),1)
                x = out
                print('cat idx:{0} \t out_idx: {1}'.format(block*self.block_depth*2+(layer*2), block*self.block_depth*2+(layer*2)+1))
                print(self.encoder[block*self.block_depth*2+(layer*2)])
                print(self.encoder[block*self.block_depth*2+(layer*2)+1])
                cat_out = self.encoder[block*self.block_depth*2+(layer*2)](x)
                out  = self.encoder[block*self.block_depth*2+(layer*2)+1](in2)
                print('block: {0} \t layer: {1} \t x: {2} \t in2:{3} \t cat_out: {4} \t out: {5}'.format(block, layer, x.shape, in2.shape, cat_out.shape, out.shape))
                print(layer)
                if layer == 4:
                    x = out
                    print("later: 4 \t x:{0}".format(x.shape))
        return out

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 5, block_depth = 5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.conv1 = nn.Conv2d(3,16,3,2)
        self.conv2 = nn.Conv2d(16,32,3,2)
        self.encoder = fibModule(in_channels = 32, num_blocks = self.num_blocks, block_depth = self.block_depth)

    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.conv2(inputs)
        return self.encoder(inputs)
# output shape = [(W−K+2P)/S]+1
"""Load Cuda """
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
""""""""""""""""""
f = FibNet(num_blocks=3,block_depth=3)
f.cuda()
# x = torch.rand(1,3,224,224)
# y = f(x)
# print(y)
summary(f,(3,224,224))
macs, params = get_model_complexity_info(f, (3, 224, 224), as_strings=True,
                                        print_per_layer_stat=True, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))