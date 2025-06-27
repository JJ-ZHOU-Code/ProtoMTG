import torch
import torch.nn as nn
import torch.nn.functional as F

import models.modules as md

class TaskHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=0)]
        model += [nn.Sigmoid()] # nn.Tanh()
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
        

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        proto_channels,
        out_channels,
        dropout = 0.0
        ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels+skip_channels+proto_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
            
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip=None, protomap=None, proto_scale_factor=2):
        x = self.model(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if protomap is not None:
            protomap = F.interpolate(protomap, scale_factor=proto_scale_factor, mode="nearest")
            x = torch.cat([x, protomap], dim=1)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        output_channels,
        proto_channels=0,
        n_blocks=5,
        skip = False,
        use_batchnorm=True,
        attention_type=None,
        center = None
    ):
        super().__init__()
        self.skip = skip 
        
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        if skip:
            skip_channels = [0] + list(encoder_channels[1:])
        else: 
            skip_channels = [0]*len(in_channels)
        proto_channels = [proto_channels] * len(skip_channels)
        out_channels = decoder_channels
        dropout = [0.5]*(len(in_channels)-2)+[0]*2

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, p_ch, out_ch, dp)
            for in_ch, skip_ch, p_ch, out_ch, dp in zip(in_channels, skip_channels, proto_channels, out_channels,dropout)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.task_head = TaskHead(
            in_channels=decoder_channels[-1],
            out_channels=output_channels,
        )

    def forward(self, features, protomap=None):
        if isinstance(features, list):
            features = features[1:]  # remove first skip with same spatial resolution
            features = features[::-1]  # reverse channels to start from head of encoder
            
            head = features[0]
            if self.skip:
                skips = features[1:]
            else: 
                skips = []
        else: 
            head = features
            skips = []
            
        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, protomap, 2**(i+1))
        
        x = self.task_head(x)
        return x