




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


