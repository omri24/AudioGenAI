import math
import torch
from torch import nn
"""
using as a start EfficientNet-B0 Architecture. The model consists of three parts:
1- stem layer - initial layer
2- body layer which consists of MBconv blocks

3- head - fully connected
"""
def calculate_output_size(input, kernel, padding, stride):
    output = (1 + math.floor((input - kernel + 2 * padding) / stride))
    if output < 1:
        raise ValueError(
            f"Invalid parameters: input={input}, kernel={kernel}, padding={padding}, stride={stride} result in negative or zero output size."
        )
    return output

class STEM_LAYER(nn.Module): #
    def __init__(self, input_size, dropout):
        super(STEM_LAYER, self).__init__()
        """
        stem layer class, architecture taken from https://medium.com/image-processing-with-python/efficientnetb0-architecture-stem-layer-496c7911a62d
        :param input_size: (in_channels, height, width)
        """
        out_channels=32
        kernel_size=(3,3)
        self.CONV1 = nn.Conv2d(kernel_size=kernel_size, in_channels=input_size[0], out_channels=out_channels, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.output_size =(out_channels, calculate_output_size(input=input_size[1], kernel=kernel_size[1], padding=0, stride=2), calculate_output_size(input=input_size[2], kernel=kernel_size[0], padding=0, stride=2) )
        self.BN = nn.BatchNorm2d(num_features=  out_channels)
        self.SWISH = nn.SiLU()
    def forward(self, x):
        x=self.CONV1(x)
        x = self.dropout(x)
        x=self.BN(x)
        x=self.SWISH(x)
        return x

class MBCONV(nn.Module):
    def __init__(self,
                 input_size,
                 expand_ratio,
                 final_output_channels,
                 kernel_size,
                 stride,
                 dropout):
        super(MBCONV, self).__init__()
        """
        :param input_size:  (input channels, height, width)
        :param expand_ratio:  for conv, if 1 skips expansion
        :param kernal_size: for conv
        :param stride: for conv

        Options to add -
        1 -  Squeeze-and-Excitation (SE) Block
        """
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)  # For SAME padding
        #Expansion
        self.expand = (expand_ratio!=1)
        output_channels = input_size[0] * expand_ratio
        if(self.expand):
            self.conv_expan = nn.Conv2d(out_channels=output_channels, in_channels=input_size[0], kernel_size=kernel_size, stride=stride, padding=self.padding)
            output_size_expansion = (output_channels,
                                     calculate_output_size(input=input_size[1], kernel=kernel_size[0], padding=self.padding[0], stride=stride),
                                     calculate_output_size(input=input_size[2], kernel=kernel_size[1], padding=self.padding[1], stride=stride))
            self.BN_expan = nn.BatchNorm2d(num_features= output_channels)
            self.SWISH_expan = nn.SiLU()
        else:
            output_size_expansion=input_size
            self.conv_expan =None
            self.BN_expan = None
            self.SWISH_expan = None
        #Depth-wise Conv
        self.DEPTH_CONV = nn.Conv2d(groups=output_channels,
                                    in_channels=output_channels,
                                    out_channels=output_channels,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    stride=stride)
        self.output_size = (final_output_channels, calculate_output_size(input=output_size_expansion[1], kernel=kernel_size[0], padding=self.padding[0], stride=stride), calculate_output_size(input=output_size_expansion[2], kernel=kernel_size[1], padding=self.padding[1], stride=stride))
        self.BN_depth=nn.BatchNorm2d(num_features=output_channels)
        self.SWISH_depth=nn.SiLU()
        self.dropout_depth = nn.Dropout2d(dropout)
        #Projection Phase
        self.projection_conv = nn.Conv2d(in_channels=output_channels, out_channels=final_output_channels, kernel_size=(1,1), stride=1, padding=0)
        self.projection_bn=nn.BatchNorm2d(num_features=final_output_channels)
        self.Swish_projection=nn.SiLU()
        print(self.output_size)
    def forward(self, x):
        identity = x
        #expand
        if(self.expand):
            x = self.conv_expan(x)
            x= self.BN_expan(x)
            x=self.SWISH_expan(x)
        #depth-wise
        x=self.DEPTH_CONV(x)
        x=self.BN_depth(x)
        x=self.SWISH_depth(x)
        x=self.dropout_depth(x)
        #projection
        x = self.projection_conv(x)
        x = self.projection_bn(x)
        x = self.Swish_projection(x)
        #skip connections
        if self.expand and identity.shape == x.shape:
          x = x + identity
        return x



class MODEL_BODY(nn.Module):
    def __init__(self, input_size, device, depth_dropout):
        super(MODEL_BODY, self).__init__()
        """
        :param input_size: in_channels, height, width)
        """
        #mbconv_size = [(1,3,3,16), (6,3,3,24), (6,5,5,40),(6,3,3,80), (6, 5,5,112), (6,3,3,192), (6,3,3,320)] #(expand_ration, kernalsize, output_size)
        #number_of_layers=[1,2,2,3,3,4,1] #number of repeats

        mbconv_size = [(1,3,3,8), (3,3,3,12), (3,3,3,16)] #(expand_ration, kernalsize, output_size)
        number_of_layers=[2,2,2] #number of repeats
        self.conv_layers=nn.ModuleList()
        for i,size in enumerate(mbconv_size):
            for n in range(number_of_layers[i]):
                new_layer=MBCONV(input_size=input_size, kernel_size=(size[1],size[2]), expand_ratio=size[0], stride=1, final_output_channels=size[3], dropout=depth_dropout).to(device)
                input_size=new_layer.output_size
                self.conv_layers.append(new_layer)
        print("number of layers", len(self.conv_layers))
        self.output_size=input_size
        self._init_weights()
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    def forward(self,x):
        for l in self.conv_layers:
            x=l.forward(x)
        return x


class HEAD(nn.Module):
    def __init__(self,input_size, output_size, dropout):
        super(HEAD, self).__init__()

        """
        :param input_size: (in_channels, height, width)
        :param output_size: sequence_len
        things to add:
        1 - dropout
        """
        kernel = (3,3)
        print(f"head input size {input_size}")
        h = calculate_output_size(input=input_size[1], kernel=kernel[0], padding=0, stride=1)
        w = calculate_output_size(input=input_size[2], kernel=kernel[1], padding=0, stride=1)
        channels = 30
        print(f"channels = {channels}, h={h}, w={w}")
        self.conv = nn.Conv2d(in_channels=input_size[0], out_channels=channels, kernel_size=kernel, stride=1)
        self.bn = nn.BatchNorm2d(num_features=channels)
        self.swish = nn.SiLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=channels*h*w, out_features=output_size*64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=output_size*64, out_features=output_size*128)
        self.output_size = output_size
        self._init_weights()
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    def forward(self,x):
        x = self.conv(x)
        x=self.bn(x)
        x=self.swish(x)
        x=self.dropout(x)
        x=self.flat(x)
        #fully connected layer 1
        x=self.fc1(x)
        x=self.swish(x)
        x=self.dropout(x)
        #fully connected layer 2
        x=self.fc2(x)
        x=torch.reshape(x, (-1, 128, self.output_size))
        return x

class CNN_MODEL (nn.Module): #using
    def __init__(self, sequence_len, device, dropout_depth = 0.1, dropout = 0):
        super(CNN_MODEL, self).__init__()

        """
        :param input_size: (in_channels, height, width)
        """
        print("setting up model")
        self.stem_layer=STEM_LAYER(input_size=(1, 128, sequence_len), dropout=dropout).to(device) #input_size: (in_channels, height, width)
        print("finished setting up stem layer")
        self.body = MODEL_BODY(input_size=self.stem_layer.output_size, device=device, depth_dropout=dropout_depth).to(device)
        print("finished setting up body")
        self.head = HEAD(input_size=self.body.output_size, output_size=sequence_len, dropout=dropout).to(device) #midi notes
        self.softmax =nn.Softmax(dim=2)
        print("finished setting up model")
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    def forward(self, x):
        x = x.unsqueeze(1)
        x=self.stem_layer(x)
        x=self.body(x)
        x=self.head(x)
        #return self.softmax(x)
        return x


