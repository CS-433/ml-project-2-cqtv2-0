import torch
import torch.nn as nn

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
    def __init__(self, in_channels, out_channels, L=2):
        super(Up, self).__init__()
        # Transposed convolution
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, L)

    def forward(self, x1, x2):
        x1 = self.up(x1)    
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, L=2, C=64, depth=4):
        """
        This class define the U-Net model.
        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output channels.
            bilinear (bool): Whether to use bilinear upsampling or transposed convolutions.
            base_c (int): Base number of channels.
            depth (int): Depth of the U-Net (number of downsamplings).
        """
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.depth = depth

        self.inc = DoubleConv(in_channels, C, L)
        self.downs = nn.ModuleList()
        for i in range(1, depth+1):
            self.downs.append(Down(C * (2**(i-1)), C * (2**i), L))

        self.up = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(Up(C * (2**i), C * (2**(i-1)), L))

        self.outc = OutConv(C, out_channels)
                
    def forward(self, x):
        """This function is used to forward pass the input through the model

        Args:
            x (torch.tensor): The input tensor

        Returns:
            torch.tensor: the predicted output
        """
        x_inc = self.inc(x)
        x_downs = [x_inc]
        for down in self.downs:
            x_down = down(x_downs[-1])
            x_downs.append(x_down)

        x = x_downs[-1]
        for i, up in enumerate(self.up):
            x = up(x, x_downs[-(i+2)])

        logits = self.outc(x)
        return nn.Sigmoid()(logits), None
