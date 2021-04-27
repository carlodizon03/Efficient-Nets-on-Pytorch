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
    def __init__(self):
        super().__init__()

    def fibonacci(self,depth):

        f = []
        for i in range(1,depth+1):
            num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
            den = 2**i * (math.sqrt(5))
            f.append(int(num/den))
        return(f)

    def naive_block_channels_variation(self, blocks, in_channels = 3,  depth = 5, ratio = 0.618):
        channel_list =[3]
        for block_idx, block in enumerate(blocks):
            depth_ = depth
            ratio_ = ratio 
            while depth_ > 0:
                val = int( (block * ratio_ * (1 - ratio_))*100)
                channel_list.append(val)
                ratio_ = 3.414 * ratio_ * (1 - ratio_)
                depth_ -= 1
        return channel_list   
        
    def forward(self, in_channels = 3, num_blocks = 5, block_depth = 5):
        initial_ratio = 0.618
        blocks_channel_list= self.naive_block_channels_variation(self.fibonacci(num_blocks),  in_channels, block_depth)
        encoder = nn.ModuleList()
        for block in range(num_blocks):
            
            in_channels = blocks_channel_list[block*block_depth]
            out_channels = blocks_channel_list[block*block_depth+1]
            encoder.append(nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3))
            # print(block, 0, in_channels,out_channels)
            for layer in range(1,block_depth):
                idx =  block*block_depth+layer
                in_channels = blocks_channel_list[idx] + blocks_channel_list[idx-1]
                out_channels = blocks_channel_list[idx+1]
                encoder.append(nn.Conv2d(in_channels,out_channels,kernel_size=3))
                # print(block, layer, in_channels,out_channels)
                if idx +1 == block_depth * num_blocks:
                    break
        return encoder

class FibNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, num_blocks = 5, block_depth = 5, ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks  = num_blocks
        self.block_depth = block_depth
        self.mainModule = fibModule()
        self.encoder = self.mainModule(self.in_channels, self.num_blocks,self.block_depth)

    def forward(self, in_tensor):
        x = in_tensor
        for block in range(self.num_blocks):
            in_ch = x 
            x2 = self.encoder[block*self.num_blocks](in_ch)
            for layer in range(1, self.block_depth):
                idx = block*block_depth+layer
                x = torch.cat((x,x2))
                x2 = self.encoder[idx](x)
f = FibNet()
f() 