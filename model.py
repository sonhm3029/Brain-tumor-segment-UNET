import torch.nn as nn
from torchvision.transforms.functional import resize
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channel=1, filters=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        
        for feature in filters:
            self.down.append(DoubleConv(
                in_channels, feature
            ))
            in_channels = feature
        
        for feature in reversed(filters):
            self.up.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2
            ))
            self.up.append(
                DoubleConv(feature*2, feature)
            )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(filters[-1], filters[-1] * 2)
        self.final_conv = nn.Conv2d(
            filters[0], out_channel, kernel_size=1
        )
    def forward(self, x):
        skip_connections = []
        
        for idx, layer in enumerate(self.down):
            x = layer(x)
            skip_connections.append(x.clone())
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]    
            
        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            skip_layer = skip_connections[idx//2]
            
            if x.shape != skip_layer.shape:
                x = resize(x, size=skip_layer.shape[2:])
            
            concat_skip = torch.cat((skip_layer, x), dim=1)
            x = self.up[idx+1](concat_skip)
            
        return self.final_conv(x)
    

            
            