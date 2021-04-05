import torch 
import torch.nn as nn
import torch.nn.functional as F 

from DConv import DConv2d

class DCNN_VGG19(nn.Module):
    def __init__(self, in_channels: int, n_class: int):
        super(VGG19_DCNN, self).__init__()

        self.in_channels = in_channels
        self.n_class = n_class

        self.model = nn.Sequential()
                                
        self._dconv2(self.in_channels, 64)  # out_ch = 64 * 5 = 320
        self._dconv2(320, 128)      # in_ch = 320, out_ch = 128 * 5 = 640
        self._dconv4(640, 256)      # in_ch = 640, out_ch = 256 * 5 = 1280
        self._dconv4(1280, 512)     # in_ch = 1280, out_ch = 512 * 5 = 2560
        self._dconv4(2560, 512)     # in_ch = 2560, out_ch = 512 * 5 = 2560

        # in_ch = 2560, out_ch = 512
        self.model.add_module('kernel_1_conv', nn.Conv2d(2560, 512, 1))
        self.model.add_module('avgpool', nn.AdaptiveAvgPool2d((7, 7)))
        self.model.add_module('flatten', nn.Flatten())

        self._fully_connected(512 * 7 * 7, self.n_class)

        self.n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
    
    def forward(self, x):
        y = self.model(x)
        return y

    def _dconv2(self, in_channels, n_filters):
        name = 'dconv2_{}_{}'.format(in_channels, n_filters * 5)
        modules = nn.Sequential(
            DConv2d(in_channels, n_filters, 3, padding=1), nn.ReLU(),
            DConv2d(n_filters * 5, n_filters, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.model.add_module(name=name, module=modules)

    def _dconv4(self, in_channels, n_filters):
        name = 'dconv4_{}_{}'.format(in_channels, n_filters * 5)
        modules = nn.Sequential(
            DConv2d(in_channels, n_filters, 3, padding=1), nn.ReLU(),
            DConv2d(n_filters * 5, n_filters, 3, padding=1), nn.ReLU(),
            DConv2d(n_filters * 5, n_filters, 3, padding=1), nn.ReLU(),
            DConv2d(n_filters * 5, n_filters, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.model.add_module(name=name, module=modules)

    def _fully_connected(self, in_features, n_class):
        name = 'fc_{}_{}'.format(in_features, n_class)
        modules = nn.Sequential(
            nn.Linear(in_features, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, n_class)
        )

        self.model.add_module(name=name, module=modules)

class DCNN_Simple(nn.Module):
    def __init__(self, in_channels: int, n_class: int):
        super(DCNN_Simple, self).__init__()

        self.in_channels = in_channels
        self.n_class = n_class

        self.model = nn.Sequential(
            DConv2d(self.in_channels, 8, 3),  # out_ch = 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  

            DConv2d(40, 16, 3),               # in_ch = 8*5 = 40, out_ch = 16  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7*7*80, self.n_class),    
        )

        self.n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

class DCNN_Medium(nn.Module):
    def __init__(self, in_channels: int, n_class: int):
        super(DCNN_Medium, self).__init__()
        
        self.in_channels = in_channels
        self.n_class = n_class

        self.model = nn.Sequential(
            DConv2d(self.in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            DConv2d(320, 64, 3, padding=1),
            nn.ReLU(),
            DConv2d(320, 128, 3, padding=1),
            nn.ReLU(),

            DConv2d(640, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            DConv2d(640, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            
            nn.Linear(7*7*640, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_class),
        )

        self.n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
    
    def forward(self, x):
        logits = self.model(x)
        return logits