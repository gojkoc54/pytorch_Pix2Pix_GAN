import torchvision
import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, scale_factor = 2, use_bn = True, use_dropout = False):
        super(UNetBlock, self).__init__()

        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        
        self.conv = self.make_conv_layer()
        if self.use_bn:
            self.bn = self.make_batchnorm()
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.activation = self.make_activation()
        

    def make_conv_layer(self):
        raise NotImplementedError

    def make_activation(self):
        raise NotImplementedError

    def make_batchnorm(self):
        raise NotImplementedError

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.activation(out)

        return out



class UNetContractingBlock(UNetBlock):
    def __init__(self, in_channels, scale_factor = 2, use_bn = True, use_dropout = False):
        super(UNetContractingBlock, self).__init__(in_channels, scale_factor, use_bn, use_dropout)

    def make_conv_layer(self):
        return nn.Conv2d(
            self.in_channels, self.scale_factor * self.in_channels, kernel_size = 4, stride = 2, padding = 1
            )

    def make_activation(self):
        return nn.LeakyReLU(negative_slope = 0.2)

    def make_batchnorm(self):
        return nn.BatchNorm2d(self.in_channels * self.scale_factor)



class UNetExpandingBlock(UNetBlock):
    def __init__(self, in_channels, scale_factor = 4, use_bn = True, use_dropout = False):
        super(UNetExpandingBlock, self).__init__(in_channels, scale_factor, use_bn, use_dropout)

    def make_conv_layer(self):
        return nn.ConvTranspose2d(
            self.in_channels, self.in_channels // self.scale_factor, kernel_size = 4, stride = 2, padding = 1
            )

    def make_activation(self):
        return nn.ReLU()
    
    def make_batchnorm(self):
        return nn.BatchNorm2d(self.in_channels // self.scale_factor)


    def forward(self, x, skip_con):

        if skip_con is not None:
            x = torch.cat([x, skip_con], axis = 1)
        out = super(UNetExpandingBlock, self).forward(x)

        return out



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels = 2, depth = 8):
        super(UNet, self).__init__()

        self.depth = depth

        self.set_hidden_channels = nn.Conv2d(in_channels, hidden_channels, kernel_size = 1)
        self.set_output_channels = nn.Conv2d(hidden_channels, out_channels, kernel_size = 1)
        
        self.contracting_blocks = nn.ModuleList()
        for i in range(self.depth):
            
            self.contracting_blocks.append(
                UNetContractingBlock(hidden_channels * 2**i, use_bn = (i != self.depth - 1))
                )

        self.expanding_blocks = nn.ModuleList()
        # First expanding block:
        self.expanding_blocks.append(
            UNetExpandingBlock(
                2 * hidden_channels * 2**(self.depth - 1), scale_factor = 2, 
                use_bn = False, use_dropout = True)
            )

        dropout_layers = [i for i in range(self.depth - 2, self.depth - 4, -1)]
        for i in range(self.depth - 2, -1, -1):
            use_dropout = i in dropout_layers

            self.expanding_blocks.append(
                UNetExpandingBlock(4 * hidden_channels * 2**i, use_bn = True, use_dropout = use_dropout) 
                )


    def forward(self, x):

        out = self.set_hidden_channels(x)
        print("set hidden - ", out.shape)

        skip_cons = []
        for i in range(self.depth):
            out = self.contracting_blocks[i](out)
            skip_cons += [out]
            print(f"contract {i} - ", out.shape)
        
        for i in range(self.depth):
            if i == 0:
                out = self.expanding_blocks[i](out, None)
            else:
                out = self.expanding_blocks[i](out, skip_cons[-i - 1])
            print(f"expand {i} - ", out.shape)
        
        out = self.set_output_channels(out)
        print("set output - ", out.shape)

        return out
        