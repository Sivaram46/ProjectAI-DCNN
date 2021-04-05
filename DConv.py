from typing import Tuple

import torch 
import torch.nn as nn
import torch.nn.functional as F 

import torchvision.transforms.functional as TF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DConv2d(nn.Module):
    __doc__ = r"""Applies 2D differential convolution over an input image 
    composed of several input plances.

    More about differential convolution at the end.

    Here, the notable parameter is `n_filters` which determines the number of 
    output channels for a given batch of inputs. First, the image is convoluted
    with a learnable filter and then appended with feature maps computed from 
    Prewitt operators for horizontal, vertical and diagonal edges. Other 
    operations are same as conventional Conv2d layer. 

    If a conv layer has n_filters as a parameter, then the output_channels is 
    5 * n_filters. Out of which input_channels * n_filters * kernel_size are 
    learnable parameters.

    References:
    ----------
    M. Sarıgül a, B.M. Ozyildirim b , M. Avci c, Differential convolutional 
    neural network, https://doi.org/10.1016/j.neunet.2019.04.025.
    """
    def __init__(
        self, in_channels: int, n_filters: int, kernel_size: Tuple[int, ...],
        stride: int=1, padding: int=0, dilation: int=1, groups: int=1,
        bias: bool=True, padding_mode: str='zeros'
    ):
        
        super(DConv2d, self).__init__()

        self.in_channels = in_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

        # Output channels for this layer 
        self.out_channels = (self.n_filters * 4) + self.n_filters

        self.conv = nn.Conv2d(
            self.in_channels, self.n_filters, self.kernel_size,
            self.stride, self.padding, self.dilation, self.groups,
            self.bias, self.padding_mode
        )

        # Prewitt operators for extra feature maps
        k1 = torch.tensor([[1, -1]]).float().view(1, 1, 1, 2)
        k2 = torch.tensor([[1], [-1]]).float().view(1, 1, 2, 1)
        k3 = torch.tensor([[1, 0], [0, -1]]).float().view(1, 1, 2, 2)
        k4 = torch.tensor([[0, 1], [-1, 0]]).float().view(1, 1, 2, 2)

        self._prewitt_op = [k1, k2, k3, k4]
        # Padding mode for each of the Prewitt operators
        self._pad_list = [
            [0, 0, 1, 0], [0, 0, 0, 1], 
            [0, 0, 1, 1], [0, 0, 1, 1]
        ]    

    def forward(self, x : torch.tensor):
        x = self.conv(x)
        f_maps = self._feature_maps(x)
        x = torch.cat((x, f_maps), dim=1)

        return x

    def _feature_maps(self, x : torch.tensor):
        maps = []
        for i in range(4):
            pad = TF.pad(x, self._pad_list[i])
            conv = F.conv2d(
                pad, 
                self._prewitt_op[i].repeat(self.n_filters, 1, 1, 1).to(DEVICE), 
                groups=self.n_filters
            )
            maps.append(conv)
            
        f_maps = torch.cat(maps, dim=1)
        return f_maps 