import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict
from ptflops import get_model_complexity_info
from Models.IRIS_Net import IRIS_Net    
from Models.IRIS_Netv2  import  IRIS_Netv2
from Models.UNet import UNet
from Models.HarDNet import HarDNet
from Models.FCHarDNet import hardnet as FCHarDNet
from Models.ENet import ENet
from Models.SegNet import SegNet
from Models.ENet_no_tConv import ENet_no_tConv
"""Load Cuda """
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
""""""""""""""""""

models = {
        "IRIS_Net": IRIS_Net,
        "IRIS_Netv2": IRIS_Netv2,
        "UNet": UNet,
        "HarDNet-DWS-39": HarDNet,
        "FCHarDNet": FCHarDNet,
        "ENet": ENet,
        "SegNet": SegNet,
        "ENet_no_tConv": ENet_no_tConv
        }

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type = str, help = "Model Version", default = "IRIS_Net")
parser.add_argument("-t", "--train_mode", type = str, help = "Model training mode.", default = "segmentation")
parser.add_argument("-in", "--input_channels", type = int, help = "Input size", default = 3)
parser.add_argument("-out", "--output_channels", type = int, help = "Output size", default = 1)
parser.add_argument("-f", "--initial_features", type = int, choices = [16,32,64], help = "Output size", default = 64)
parser.add_argument("-img_size","--image_size", type = int , help = "Image size.", default = 224)
parser.add_argument("-s", "--save", type = bool, help = "Save Summary", default = True)
parser.add_argument("-o", "--out_path", type = str, help = "Save path", default = "Models/profiles/")
args = parser.parse_args()

ch_in = args.input_channels
ch_out = args.output_channels
img_size = args.image_size
initial_features = args.initial_features
model = args.model
train_mode = args.train_mode
is_save = args.save
save_path = args.out_path


if(model == "IRIS_Net"):
    net = models[model](in_channels = ch_in, out_channels = ch_out, initial_features = initial_features, train_mode = train_mode)
elif(model == "IRIS_Netv2"):
    net = models[model](in_channels = ch_in, out_channels = ch_out, initial_features = initial_features)
elif(model == "UNet"):
    net = models[model](in_channels = ch_in, out_channels = ch_out, init_features = initial_features)
elif(model == "HarDNet-DWS-39"):
    net = models[model](depth_wise = True, arch = 39, pretrained = False)
elif(model == "FCHarDNet"):
    net = models[model]()
elif(model == "ENet"):
    net = models[model](num_classes = 1)
elif(model == "SegNet"):
    net = models[model](n_classes = 1)
elif(model == "ENet_no_tConv"):
    net = models[model](num_classes = 1)


net.to(device)
su = str(summary(net,(ch_in, img_size, img_size)))
su_list = list(su.split('\n'))
macs, _ = get_model_complexity_info(net, (3,224,224), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8} / {} GFLOPS'.format('Computational complexity: ', macs, float(macs[:4])*2))
su_list.append('{:<30}  {:<8} / {} GFLOPS'.format('Computational complexity: ', macs, float(macs[:4])*2))
su_list.append("==========================================================================================")

import time
import numpy as np
times = []

for i in range(15):
    net.eval()
    ins = torch.rand([1,ch_in,img_size,img_size]).to(device)
    start = time.time()
    preds = net(ins)
    tlapse = time.time() - start
    times.append(tlapse)
    print("Image {0} Inference Time: {1:.4f}".format(i,tlapse))
    su_list.append("Image {0} Inference Time: {1:.4f}".format(i,tlapse))

print("Mean Inference time per image: {0:.4f}".format(np.mean(times)))
su_list.append("Mean Inference time per image: {0:.4f}".format(np.mean(times)))

print("==========================================================================================")
su_list.append("==========================================================================================")

if is_save:
    file = open(save_path+model+'_'+train_mode+'_'+str(ch_in)+'_'+str(ch_out)+'_'+str(initial_features)+'.txt', "w")
    for val in su_list:
        file.write(val)
        file.write('\n')
    file.close()