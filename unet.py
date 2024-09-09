# Implement your UNet model here

# assert False, "Not implemented yet!"
import torch
import torch.nn as nn
from torchinfo import summary

class double_conv(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.db_conv=nn.Sequential(
                nn.Conv2d(input,output,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True),
                nn.Conv2d(output,output,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True)
                )
    def forward(self,x):
        return self.db_conv(x)
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.down1 = double_conv(3,64)
        self.down2 = double_conv(64,128)
        self.down3 = double_conv(128,256)
        self.down4 = double_conv(256,512)

        self.bottleneck = double_conv(512,1024) 

        self.softmax = nn.Softmax()

        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.upsample1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.upsample2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.upsample3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.upsample4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        
        self.up1 = double_conv(1024, 512)
        self.up2 = double_conv(512, 256)
        self.up3 = double_conv(256,128)
        self.up4 = double_conv(128,64) 
        
        self.out = nn.Conv2d(64,1,kernel_size=1)
    
    
    def forward(self,x):
        x1 = self.down1(x)
        x1_down = self.maxpool(x1)

        x2 = self.down2(x1_down)
        x2_down = self.maxpool(x2)

        x3 = self.down3(x2_down)
        x3_down = self.maxpool(x3)
        
        x4 = self.down4(x3_down)
        x4_down = self.maxpool(x4)
        
        x5 = self.bottleneck(x4_down)

        x6_up = self.upsample1(x5)
        mix1 = torch.cat([x6_up, x4], dim=1)
        x6 = self.up1(mix1)
        
        x7_up = self.upsample2(x6)
        mix2 = torch.cat([x7_up, x3], dim=1)
        x7 = self.up2(mix2)
        
        x8_up = self.upsample3(x7)
        mix3 = torch.cat([x8_up, x2], dim=1)
        x8 = self.up3(mix3)
        
        x9_up = self.upsample4(x8)
        mix4 = torch.cat([x9_up, x1], dim=1)
        x9 = self.up4(mix4)
        
        out = self.out(x9)

        return out

# print(Unet().eval())