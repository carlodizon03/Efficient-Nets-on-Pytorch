import torch.nn as nn
from torch.autograd import Variable

class conv2DBatchNormRelu(nn.Module):
	def __init__(
	    self,
	    in_channels,
	    n_filters,
	    k_size,
	    stride,
	    padding,
	    bias=True,
	    dilation=1,
	    with_bn=True,
	):
	    super(conv2DBatchNormRelu, self).__init__()

	    conv_mod = nn.Conv2d(
	    	int(in_channels),
	    	int(n_filters),
	    	kernel_size=k_size,
	    	padding=padding,
	    	stride=stride,
	    	bias=bias,
	    	dilation=1,
	    	)

	    self.cbr_unit = nn.Sequential(
	    	conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
	    	)

	def forward(self, inputs):
	    outputs = self.cbr_unit(inputs)
	    return outputs

class encoder1(nn.Module):
	def __init__(self, in_size, out_size):
		super(encoder1, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool2d = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		unpooled_shape = x.size()
		x, ind = self.maxpool2d(x)
		return x, ind, unpooled_shape

class encoder2(nn.Module):
	def __init__(self, in_size, out_size):
		super(encoder2, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool2d = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		unpooled_shape = x.size()
		x, ind = self.maxpool2d(x)
		return x, ind, unpooled_shape

class decoder1(nn.Module):
	def __init__(self, in_size, out_size):
		super(decoder1, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, x, ind, out_shape):
		outputs = self.unpool(input=x, indices=ind, output_size=out_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		return outputs			

class decoder2(nn.Module):
	def __init__(self, in_size, out_size):
		super(decoder2, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, x, ind, out_shape):
		outputs = self.unpool(input=x, indices=ind, output_size=out_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		return outputs

        
class SegNet(nn.Module):
    """autoencoder definition
    """
    def __init__(self, n_classes=1, in_channels=3, is_unpooling=True):
        super(SegNet, self).__init__()
        
        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.enc1 = encoder1(self.in_channels, 64)
        self.enc2 = encoder1(64, 128)
        self.enc3 = encoder2(128, 256)
        self.enc4 = encoder2(256, 512)
        self.enc5 = encoder2(512, 512)

        self.dec5 = decoder2(512, 512)
        self.dec4 = decoder2(512, 256)
        self.dec3 = decoder2(256, 128)
        self.dec2 = decoder1(128, 64)
        self.dec1 = decoder1(64, n_classes)


    def forward(self, x):

        enc1, ind_1, unpool_sh1 = self.enc1(x)
        enc2, ind_2, unpool_sh2 = self.enc2(enc1)
        enc3, ind_3, unpool_sh3 = self.enc3(enc2)
        enc4, ind_4, unpool_sh4 = self.enc4(enc3)
        enc5, ind_5, unpool_sh5 = self.enc5(enc4)

        dec5 = self.dec5(enc5, ind_5, unpool_sh5)
        dec4 = self.dec4(dec5, ind_4, unpool_sh4)
        dec3 = self.dec3(dec4, ind_3, unpool_sh3)
        dec2 = self.dec2(dec3, ind_2, unpool_sh2)
        dec1 = self.dec1(dec2, ind_1, unpool_sh1)

        return dec1

    def vgg16_init(self, vgg16):
        covNets = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]

        vgg16Features = list(vgg16.features.children())

        vgg16Layers = []
        for _layer in vgg16Features:
            if isinstance(_layer, nn.Conv2d):
                vgg16Layers.append(_layer)

        outLayers = []
        for idx, conv in enumerate(covNets):
            if idx < 2:
                layers = [conv.conv1.cbr_unit, conv.conv2.cbr_unit]
            else:
                layers = [
                    conv.conv1.cbr_unit,
                    conv.conv2.cbr_unit,
                    conv.conv3.cbr_unit
                ]
            for lay in layers:
                for _layer in lay:
                    if isinstance(_layer, nn.Conv2d):
                        outLayers.append(_layer)

        assert len(vgg16Layers) == len(outLayers)

        for xx, yy in zip(vgg16Layers, outLayers):
            if isinstance(xx, nn.Conv2d) and isinstance(yy, nn.Conv2d):
                assert xx.weight.size() == yy.weight.size()
                assert xx.bias.size() == yy.bias.size()
                yy.weight.data = xx.weight.data
                yy.bias.data = xx.bias.data
