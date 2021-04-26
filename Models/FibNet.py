from collections import OrderedDict
import torch
import torch.nn as nn
import math
from torchsummary import  summary
from collections import defaultdict
# from ptflops import get_model_complexity_info





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
    def __init__(self, in_channels, depth):
        super().__init__()

        self.in_channels = in_channels
        self.depth = depth

    
    def fibonacci(self,depth):

        f = []
        for i in range(1,depth+1):
            num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
            den = 2**i * (math.sqrt(5))
            f.append(int(num/den))
        return(f)

    def naive_block_channels_variation(self, blocks, depth = 5, ratio = 0.618):
        channel_list = defaultdict(dict)
        for block_idx, block in enumerate(blocks):
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list['block_%d'%block_idx]['layer_%d'%(depth-depth_)] = val
                ratio_ = 3.414 * ratio_ * (1 - ratio_)
                depth_ -= 1
        return channel_list   
        
    def forward(self, in_channels = 3, num_blocks = 5, block_depth = 5):
        initial_ratio = 0.618
        blocks_channel_dict= self.naive_block_channels_variation(self.fibonacci(num_blocks), block_depth)
        encoder = nn.ModuleList()
        previous_out_channels = 0
        for block in range(num_blocks):
            for layer in range(block_depth):
                blk = 'block_%'%block
                lyr = 'layer_%'%layer
                in_channels = in_channels + previous_out_channels
                out_channels = blocks_channel_dict[blk][lyr]
                encoder[block].append(nn.Conv2d(in_channels,out_channels,kernel_size=3))
                previous_out_channels = out_channels
        for i, d in blocks_channel_list.items():
            for i2 , d2 in d.items():
                print(i,i2,d2)
f = fibModule(3, 5)
f()    
# class FibNet(nn.Module):
#     def __init__(self, in_channels = 3, out_channels = 1, block_depth = 5):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.block_depth = block_depth
#         self.encoder = fibModule(in_channels,block_depth)

#     def forward(x):

