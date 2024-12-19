import torch
import torch.nn as nn
import gc

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, L=2):
        super(DoubleConv, self).__init__()
        blocks = []
        for i in range(L):
            block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
            blocks.append(block)
            in_channels = out_channels
        self.double_conv = nn.Sequential(*blocks)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, L=2):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, L)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, n, L=2):
        super(Up, self).__init__()
        # Transposed convolution
        self.n = n
        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(n*out_channels, out_channels, L)

    def forward(self, to_up, xs):
        x = self.up(to_up)
        xs.append(x)
        return self.conv(torch.cat(xs, dim=1))

class OutConv(nn.Module):
    """Final convolution layer"""
    def __init__(self, in_channels, out_channels):
        """Constructor for the OutConv class

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of the OutConv class

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        return self.conv(x)

class NestedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, L=2, C=64, depth=4):
        """
        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output channels.
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions.
            base_c (int): Base number of channels.
            depth (int): Depth of the U-Net (number of downsamplings).
        """
        super(NestedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth

        self.inc = DoubleConv(n_channels, C, L)

        self.down00 = Down(C, 2*C, L)
        self.down10 = Down(2*C, 4*C, L)
        self.down20 = Down(4*C, 8*C, L)
        self.down30 = Down(8*C, 16*C, L)
        
        self.layerDown31 = Up(16*C, 8*C, 2, L)

        self.layerDown21 = Up(8*C, 4*C, 2, L)
        self.layerDown22 = Up(8*C, 4*C, 3, L)
        
        self.layerDown11 = Up(4*C, 2*C, 2, L)
        self.layerDown12 = Up(4*C, 2*C, 3, L)
        self.layerDown13 = Up(4*C, 2*C, 4, L)
        
        self.layerDown01 = Up(2*C, C, 2, L)
        self.layerDown02 = Up(2*C, C, 3, L)
        self.layerDown03 = Up(2*C, C, 4, L)
        self.layerDown04 = Up(2*C, C, 5, L)

        
        self.outc = OutConv(C, n_classes)

    def forward(self, x):
        """Forward pass of the NestedUNet class

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        x_00 = self.inc(x)
        x_10 = self.down00(x_00)
        x_20 = self.down10(x_10)
        x_30 = self.down20(x_20)
        x_40 = self.down30(x_30)

        x_31 = self.layerDown31(x_40, [x_30])      
        del x_40
        gc.collect()
        x_21 = self.layerDown21(x_30, [x_20])
        del x_30  
        gc.collect()     
        x_22 = self.layerDown22(x_31, [x_20, x_21])
        del x_31
        gc.collect()
        
        x_11 = self.layerDown11(x_20, [x_10])
        del x_20
        gc.collect()
        x_12 = self.layerDown12(x_21, [x_10, x_11])
        del x_21
        gc.collect()
        x_13 = self.layerDown13(x_22, [x_10, x_11, x_12])
        del x_22
        gc.collect()
        
        x_01 = self.layerDown01(x_10, [x_00])
        del x_10
        gc.collect()    
        x_02 = self.layerDown02(x_11, [x_00, x_01])
        del x_11    
        gc.collect() 
        x_03 = self.layerDown03(x_12, [x_00, x_01, x_02])
        del x_12
        gc.collect()
        x_04 = self.layerDown04(x_13, [x_00, x_01, x_02, x_03])
        del x_13, x_00
        gc.collect()
        logits = self.outc(x_04)
        aux = [self.outc(x_01), self.outc(x_02),  self.outc(x_03), logits]

        return nn.Sigmoid()(logits), [nn.Sigmoid()(x) for x in aux]