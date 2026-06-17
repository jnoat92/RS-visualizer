""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, features=32):
        super(UNet, self).__init__()
        self.net_name = 'Unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, features)
        self.down1 = Down(features  , features*2)
        self.down2 = Down(features*2, features*4)
        self.down3 = Down(features*4, features*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features*8, features*16 // factor)
        self.up1 = Up(features*16, features*8 // factor, bilinear)
        self.up2 = Up(features*8 , features*4 // factor, bilinear)
        self.up3 = Up(features*4 , features*2 // factor, bilinear)
        self.up4 = Up(features*2 , features, bilinear)
        self.drop = nn.Dropout2d(p=0.2)
        self.outc = OutConv(features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.drop(x)
        feat = x
        
        logits = self.outc(x)
                
        return logits, feat
