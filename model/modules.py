




import os
import json
import copy
import math

from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################## About Inputs #########################
### from DataLoader
### ids, raw_texts, speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, pitches, energies, durations

### Actually Inputs of fastspeech2: batch[2:] from DataLoader
# (from dataloader): speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, pitches, energies, duration
# (inputs of v.a.) :speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len, p_targets, e_targets, d_targets, (p_control=1.0, d_control=1.0, e_control=1.0)


########################################## 08.21(Mon) ################################################
class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"] # 256
        self.filter_size = model_config["variance_predictor"]["filter_size"] # 256
        self.kernel = model_config["variance_predictor"]["kernel_size"] # 3
        self.conv_output_size = model_config["variance_predictor"]["filter_size"] #256
        self.dropout = model_config["variance_predictor"]["dropout"] # .5

        self.conv_layer = nn.Sequential(OrderedDict([
            ### 1
            ("conv1d_1", Conv(self.input_size, self.filter_size, kernel_size=self.kernel, padding=(self.kernel - 1) // 2),),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            #### 2
            ("conv1d_2", Conv(self.filter_size, self.filter_size, kernel_size=kernel, padding=(self.kernel - 1) // 2),),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout)),
                            ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out
    

########################################## 08.21(Mon) ################################################
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init="linear",):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,             ## input_size = 256
            out_channels,            ## filter_size = 256 
            kernel_size=kernel_size, ## kernel = 3
            stride=stride,           ## 1
            padding=padding,         ## 0 but (self.kernel(=3) - 1) // 2 --> 1
            dilation=dilation,       ## 1
            bias=bias,               ## True
        )

    def forward(self, x):
        # x: enc_output
        # x: [Batch Size, Max_Mel_Length, d_model] [16, 1404, 256])
        # print(x.shape)
        x = x.contiguous().transpose(1, 2)
        # x: --> [Batch Size, d_model, Max_Mel_Length,] [16, 256, 1404]
        # print(x.shape)
        x = self.conv(x)
        # x: --> [Batch Size, d_model, Max_Mel_Length,] [16, 256, 1404]
        # print(x.shape)
        x = x.contiguous().transpose(1, 2)
        # print(x.shape)
        # x: --> [Batch Size, Max_Mel_Length, d_model] [16, 1404, 256]) 

        return x


########################################## 08.20(Sun) ################################################
class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        ## x: enc_output from Encoder # (torch.Size([16, 85, 256])
        ## duration: durations from DataLoader  ## d_target in Varaince Adaptor # torch.Size([16, 203])
        ## max_len =  max_mel_len ## from DataLoader -> LR (max_len) # 1465

        output = list()
        mel_len = list()

        for batch, expand_target in zip(x, duration):
            # print(batch.shape, expand_target.shape) ## torch.Size([85, 256]) torch.Size([203])
            # expand_target: [8, 4, 7, ...] from duration: [16, 203]

            expanded = self.expand(batch, expand_target)
            # print(expanded.shape) ## [598, 256]

            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            ## output: [16]; [598, 256]
            output = pad(output, max_len)
            ## output: [16, 1465, 256]  
        else:
            output = pad(output)

        ## output: [16, 1465, 256]
        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        ## batch: expanded: [598, 256]
        ## expand_target: [8, 4, 7, ...] from duration: [16, 203]
        out = list()

        for i, vec in enumerate(batch):
            ## vec: [256]
            expand_size = predicted[i].item() ## int: 8, ...
            # vec.expand(max(int(expand_size), 0), -1) ## [256] --> [8, 256]
            out.append(vec.expand(max(int(expand_size), 0), -1)) ## [[8, 256], ...]

        ## Concat!
        out = torch.cat(out, 0)
        out.shape ## [598, 256]
        return out

    def forward(self, x, duration, max_len):
        ## x: enc_output from Encoder # (torch.Size([16, 85, 256])
        ## duration: durations from DataLoader  ## d_target in Varaince Adaptor # torch.Size([16, 203])
        ## max_len =  max_mel_len ## from DataLoader -> LR (max_len) # 1465
      
        output, mel_len = self.LR(x, duration, max_len)
        ## output: [16, 1465, 256]
        ## mel_len
        return output, mel_len


