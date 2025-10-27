import torch
from torch import nn
from typing import Optional, Union, Tuple

class ConvLayer2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
    ):
        super().__init__()  
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
        )
        if use_norm:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d(out_channels)
            layers.append(norm_layer)
        if use_act:
            if act_layer is None:
                act_layer = nn.ReLU(inplace=True)
            layers.append(act_layer)
        
     
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

 