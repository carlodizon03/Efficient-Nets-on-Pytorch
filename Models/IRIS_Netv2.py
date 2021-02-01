from collections import OrderedDict
import torch
import torch.nn as nn
class IRIS_Netv2(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, init_features = 32, is_depthwise = False):
        super(IRIS_Netv2, self).__init__()

        self.dropout = nn.Dropout(.2)
        self.init_features  = init_features
        self.in_channels    = in_channels 
        self.out_channels   = out_channels
        self.is_depthwise   = is_depthwise
        self.encoder1       = IRIS_Netv2._block(in_channels = self.in_channels, out_channels = self.init_features, block_name = "Encoder1", is_depthwise = self.is_depthwise)
        self.pool1          = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2       = IRIS_Netv2._block(in_channels = self.init_features, out_channels = self.init_features * 2, block_name = "Encoder2", is_depthwise = self.is_depthwise)
        self.pool2          = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3       = IRIS_Netv2._block(in_channels = self.init_features * 2, out_channels = self.init_features * 4, block_name = "Encoder3", is_depthwise = self.is_depthwise)
        self.pool3          = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4       = IRIS_Netv2._block(in_channels = self.init_features * 4, out_channels = self.init_features * 8,  block_name = "Encoder4", is_depthwise = self.is_depthwise)
        self.pool4          = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck     = IRIS_Netv2._block(in_channels = self.init_features * 8, out_channels = self.init_features * 16, block_name = "Bottleneck", is_depthwise = self.is_depthwise)

        self.upconv4        = nn.ConvTranspose2d(in_channels = self.init_features * 16, out_channels = self.init_features * 8, kernel_size = 2, stride = 2)
        self.decoder4       = IRIS_Netv2._block(in_channels = (self.init_features * 8) * 2, out_channels = self.init_features * 8, block_name = "Decoder4", is_depthwise = self.is_depthwise)
        self.upconv3        = nn.ConvTranspose2d(in_channels= self.init_features * 8, out_channels = self.init_features * 4, kernel_size = 2 , stride = 2)
        self.decoder3       = IRIS_Netv2._block(in_channels = (self.init_features * 4) * 2, out_channels = self.init_features * 4, block_name = "Decoder3", is_depthwise = self.is_depthwise)
        self.upconv2        = nn.ConvTranspose2d(in_channels = self.init_features * 4, out_channels = self.init_features * 2, kernel_size =2, stride = 2)
        self.decoder2       = IRIS_Netv2._block(in_channels = (self.init_features * 2) *2, out_channels = self.init_features * 2, block_name = "Decoder2", is_depthwise = self.is_depthwise)
        self.upconv1        = nn.ConvTranspose2d(in_channels = self.init_features * 2, out_channels = self.init_features, kernel_size = 2, stride = 2)
        self.decoder1       = IRIS_Netv2._block(in_channels = self.init_features * 2, out_channels = self.init_features, block_name = "Decoder1", is_depthwise = self.is_depthwise)
        self.classifier     = nn.Conv2d(in_channels = self.init_features, out_channels = self.out_channels, kernel_size = 1)

    def forward(self, x):
        
        enc1 = self.encoder1(x)
        self.dropout(enc1)
        enc2 = self.encoder2(self.pool1(enc1))
        self.dropout(enc2)
        enc3 = self.encoder3(self.pool2(enc2))
        self.dropout(enc3)
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        self.dropout(bottleneck)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4,enc4), dim = 1)
        dec4 = self.decoder4(dec4)
        self.dropout(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3,enc3), dim = 1)
        dec3 = self.decoder3(dec3)
        self.dropout(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2,enc2), dim = 1)
        dec2 = self.decoder2(dec2)
        self.dropout(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1,enc1), dim = 1)
        dec1 = self.decoder1(dec1)
        self.dropout(dec1)


        return torch.sigmoid(self.classifier(dec1))
    
    @staticmethod
    def _block(in_channels, out_channels, block_name, is_depthwise = False):

        if is_depthwise:
            return nn.Sequential(

                                OrderedDict(
                                    [
                                        (
                                            block_name + "_dw_conv", nn.Conv2d(
                                                                                in_channels  = in_channels, 
                                                                                out_channels = in_channels,
                                                                                kernel_size  = 3,
                                                                                padding      = 1,
                                                                                groups        = in_channels,
                                                                                bias         = False
                                                                            )
                                        ),
                                        (
                                            block_name + '_bnorm1', nn.BatchNorm2d(num_features = in_channels)
                                        ),
                                        (
                                            block_name + "_relu1", nn.ReLU(inplace = True) 
                                        ),
                                        (
                                            block_name + "_pw_conv", nn.Conv2d(
                                                                                in_channels  = in_channels,
                                                                                out_channels = out_channels,
                                                                                kernel_size  = 3,
                                                                                padding      = 1,
                                                                                bias         = False
                                                                            )
                                        ),
                                        (
                                            block_name + "_bnorm2", nn.BatchNorm2d(num_features = out_channels)
                                        ),
                                        (
                                            block_name + "_relu2", nn.ReLU(inplace = True)
                                        ),
                                    ]
                                )

                            )

        else:
            return nn.Sequential(

                                OrderedDict(
                                    [
                                        (
                                            block_name + "_conv1", nn.Conv2d(
                                                                                in_channels  = in_channels, 
                                                                                out_channels = out_channels,
                                                                                kernel_size  = 3,
                                                                                padding      = 1,
                                                                                bias         = False
                                                                            )
                                        ),
                                        (
                                            block_name + '_bnorm1', nn.BatchNorm2d(num_features = out_channels)
                                        ),
                                        (
                                            block_name + "_relu1", nn.ReLU(inplace = True) 
                                        ),
                                        (
                                            block_name + "_conv2", nn.Conv2d(
                                                                                in_channels  = out_channels,
                                                                                out_channels = out_channels,
                                                                                kernel_size  = 3,
                                                                                padding      = 1,
                                                                                bias         = False
                                                                            )
                                        ),
                                        (
                                            block_name + "_bnorm2", nn.BatchNorm2d(num_features = out_channels)
                                        ),
                                        (
                                            block_name + "_relu2", nn.ReLU(inplace = True)
                                        ),
                                    ]
                                )

                            )
