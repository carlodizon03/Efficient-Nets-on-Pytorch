from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import  summary
from ptflops import get_model_complexity_info

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise    = nn.Sequential(
                                        nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, groups = in_channels, bias = False),
                                        nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace = True)
                                        
                                        )
        self.pointwise = nn.Sequential(
                                        nn.Conv2d(in_channels, out_channels , kernel_size = 1),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace = True)
                                      )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
   
class IRIS_Net(nn.Module):
    def __init__(self,
                 in_channels =3 , 
                 out_channels = 1,
                 initial_features = 64, 
                 train_mode = "segmentation"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = initial_features
        
        self.dropout1   = nn.Dropout(0.05)
        self.dropout2   = nn.Dropout(0.3)

        self.train_mode = train_mode
        #encoder block1
        self.encoder1 =  nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.in_channels, out_channels = self.features),
                                        DepthWiseSeparableConv(in_channels = self.features, out_channels = self.features)
                                        )   
        self.downsample1 = nn.MaxPool2d(kernel_size = 2, stride = 2)                                  
        
        #encoder block1
        self.encoder2 =  nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features , out_channels = self.features * 2),
                                        DepthWiseSeparableConv(in_channels = self.features * 2, out_channels = self.features * 2),
                                        )   
        self.downsample2 = nn.MaxPool2d(kernel_size = 2, stride = 2)    
        
        #encoder block3
        self.encoder3 =  nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features * 2, out_channels = self.features * 4),
                                        DepthWiseSeparableConv(in_channels = self.features * 4, out_channels = self.features * 4)

                                        )   
        self.downsample3 = nn.MaxPool2d(kernel_size = 2, stride = 2)    

        #encoder block4
        self.encoder4 =  nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features * 4, out_channels = self.features * 8),
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 8)
                                        )   
        self.downsample4 = nn.MaxPool2d(kernel_size = 2, stride = 2) 

        #bottlenck block 1
        self.bottleneck1  = nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 16),
                                        DepthWiseSeparableConv(in_channels = self.features * 16, out_channels = self.features * 8),
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 16)
                                        )
        #bottlenck block 2
        self.bottleneck2  = nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features * 16, out_channels = self.features * 8),
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 8),
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 16)
                                        )
        #bottlenck block 3
        self.bottleneck3  = nn.Sequential(
                                        DepthWiseSeparableConv(in_channels = self.features * 16, out_channels = self.features * 8),
                                        DepthWiseSeparableConv(in_channels = self.features * 8, out_channels = self.features * 16),
                                        DepthWiseSeparableConv(in_channels = self.features * 16, out_channels = self.features * 8)

                                        ) 
                                                           
        #decoder block4
        self.upConv4      = nn.ConvTranspose2d(in_channels = self.features * 8, out_channels = self.features * 4, kernel_size = 2, stride = 2)
        self.decoder4     = nn.Sequential(
                                         DepthWiseSeparableConv(in_channels = (self.features * 4) * 3, out_channels = self.features * 4),
                                         DepthWiseSeparableConv(in_channels = self.features * 4, out_channels = self.features * 4)
                                        )
        #decoder block3
        self.upConv3      = nn.ConvTranspose2d(in_channels = self.features * 4, out_channels = self.features * 2, kernel_size = 2, stride = 2)
        self.decoder3     = nn.Sequential(
                                    DepthWiseSeparableConv(in_channels = (self.features * 2) * 3, out_channels = self.features * 2),
                                    DepthWiseSeparableConv(in_channels = self.features * 2, out_channels = self.features * 2)
                                    )
        #decoder block2
        self.upConv2      = nn.ConvTranspose2d(in_channels = self.features * 2, out_channels = self.features, kernel_size = 2, stride = 2)
        self.decoder2     = nn.Sequential(
                                    DepthWiseSeparableConv(in_channels = (self.features) * 3, out_channels = self.features ),
                                    DepthWiseSeparableConv(in_channels = self.features , out_channels = self.features )
                                    )
        #decoder block1
        self.upConv1       = nn.ConvTranspose2d(in_channels = self.features , out_channels = self.features // 2, kernel_size = 2, stride = 2)
        self.decoder1      = nn.Sequential(
                                    DepthWiseSeparableConv(in_channels = (self.features // 2) * 3, out_channels = (self.features // 2) ),
                                    DepthWiseSeparableConv(in_channels = (self.features // 2) , out_channels = (self.features // 2) )
                                    )

        #segmentation classifier block
        self.conv           =  nn.Sequential(     
                                            nn.Conv2d(in_channels = (self.features // 2) , out_channels = (self.features // 2), kernel_size = 3, stride = 1, padding = 1, groups = (self.features // 2), bias = False),
                                            nn.Conv2d(in_channels = (self.features // 2), out_channels = self.out_channels , kernel_size = 1)
                                        )   

        #classification classifier blocks
        self.ConvClassifier     = nn.Sequential(    
                                            nn.Conv2d(in_channels = 128, out_channels= 64, kernel_size= 3, stride= 2),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels = 64, out_channels= 32, kernel_size= 3, stride= 2),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU()
                                            )    
        self.LinearClassifier = nn.Sequential(
                                                nn.Linear(in_features = 32*13*13, out_features= 2048),
                                                nn.Linear(in_features= 2048, out_features = self.out_channels),
                                            )
    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc1 = self.dropout1(enc1)

        enc2 = self.encoder2(self.downsample1(enc1))
        enc2 = self.dropout1(enc2)

        enc3 = self.encoder3(self.downsample2(enc2))
        enc3 = self.dropout1(enc3)

        enc4 = self.encoder4(self.downsample3(enc3))
        enc4 = self.dropout1(enc4)

        bottleneck = self.downsample4(enc4)
        bottleneck = self.dropout1(bottleneck)
        
        x = self.bottleneck1(bottleneck)
        x = self.dropout1(x)
        x = self.bottleneck2(x)
        x = self.dropout1(x)
        x = self.bottleneck3(x)
        x = self.dropout1(x)
        
        dec4 = self.upConv4(x)
        dec4 = self.dropout1(dec4)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout1(dec4)

        dec3 = self.upConv3(dec4)
        dec3 = self.dropout1(dec3)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.decoder3(dec3)
        dec3 = self.dropout1(dec3)
        if(self.train_mode == "segmentation"):
            dec2 = self.upConv2(dec3)
            dec2 = self.dropout1(dec2)
            dec2 = torch.cat((dec2, enc2), dim = 1)
            dec2 = self.decoder2(dec2)
            dec2 = self.dropout1(dec2)

            dec1 = self.upConv1(dec2)
            dec1 = self.dropout1(dec1)
            dec1 = torch.cat((dec1, enc1), dim = 1)
            dec1 = self.decoder1(dec1)
            dec1 = self.dropout2(dec1)
        
            x    = self.conv(dec1)

        elif(self.train_mode == "classification"):

            x = dec3.view(-1,128*56*56)
            x = self.ConvClassifier(dec3)
            x = self.dropout2(x)
            x = x.view(-1, 32*13*13)
            x = self.LinearClassifier(x)

        return x



