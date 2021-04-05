import torch 
import torch.nn as nn
import torch.nn.functional as F 

from DConv import DConv2d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class VGG19_DCNN(nn.Module):
    def __init__(self, n_class: int, input_shape: int=None):
        super(VGG19_DCNN, self).__init__()

        self.n_class = n_class

        self.model = nn.Sequential()

        self._dconv2(3, 64)         # in_ch = 3, out_ch = 64 * 5 = 320

        self._dconv2(320, 128)      # in_ch = 320, out_ch = 128 * 5 = 640

        self._dconv4(640, 256)      # in_ch = 640, out_ch = 256 * 5 = 1280

        self._dconv4(1280, 512)     # in_ch = 1280, out_ch = 512 * 5 = 2560

        self._dconv4(2560, 512)     # in_ch = 2560, out_ch = 512 * 5 = 2560

        self.model.add_module('avgpool', nn.AdaptiveAvgPool2d((7, 7)))

        self.model.add_module('flatten', nn.Flatten())

        self._fully_connected(2560 * 7 * 7, n_class)

        self.model.to(DEVICE)
    
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
            nn.Linear(in_features, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_class)
        )

        self.model.add_module(name=name, module=modules)