import torch
from torch import nn


class UNetContractingBlock(nn.Module):
    
    def __init__(self, in_channels, use_bn = True, use_dropout = False):
        super(UNetContractingBlock, self).__init__()
        
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.conv = nn.Conv2d(
            self.in_channels, 2 * self.in_channels, 
            kernel_size = 4, stride = 2, padding = 1
            )
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(2 * self.in_channels)

        if self.use_dropout:
            self.dropout = nn.Dropout()

        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x):

        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.activation(out)

        return out
    


class UNetExpandingBlock(nn.Module):

    def __init__(self, in_channels, use_bn = True, use_dropout = False):
        super(UNetExpandingBlock, self).__init__()

        self.in_channels = in_channels
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.conv = nn.ConvTranspose2d(
            2 * self.in_channels, self.in_channels // 2, 
            kernel_size = 4, stride = 2, padding = 1
            )
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.in_channels // 2)

        if self.use_dropout:
            self.dropout = nn.Dropout()

        self.activation = nn.ReLU() 
 

    def forward(self, x, skip_con_x):
        
        out = torch.cat([x, skip_con_x], axis=1)
        
        out = self.conv(out)
        if self.use_bn:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.activation(out)

        return out



class UNet(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64, depth=4):
        super(UNet, self).__init__()

        self.depth = depth

        self.set_hidden_channels = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)

        self.contracting_layers = nn.ModuleList()
        contracting_scales = [2**i for i in range(self.depth)]
        for i in range(self.depth):
            scale = contracting_scales[i]

            self.contracting_layers.append(
                UNetContractingBlock(hidden_channels * scale)
                )

        self.expanding_layers = nn.ModuleList()
        dropout_layers = [i for i in range(3)]
        expanding_scales = [2**i for i in range(1, self.depth + 1)][ : : -1]
        for i in range(self.depth):
            scale = expanding_scales[i]
            use_dropout = i in dropout_layers
            
            self.expanding_layers.append(
                UNetExpandingBlock(hidden_channels * scale, use_dropout=use_dropout)
                )

        self.set_output_channels = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)


    def forward(self, x):

        out = self.set_hidden_channels(x)

        skip_cons = []
        for i in range(self.depth):
            out = self.contracting_layers[i](out)
            skip_cons.append(out)

        for i in range(self.depth):
            out = self.expanding_layers[i](out, skip_cons[-i - 1])

        out = self.set_output_channels(out)

        return out
