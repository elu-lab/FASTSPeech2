### Github[transformer/Layers.py]: https://github.com/ming024/FastSpeech2/blob/master/transformer/Layers.py



from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward

### 할 게 없어지네..?

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        ## enc_output: [16, 90, 256]

        ## mask when Encoder: src_mask
        ## mask Shape: [16, 90,]
        ## mask.unsqueeze(-1) Shape:[16, 90, 1]
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        
        enc_output = self.pos_ffn(enc_output)

        ## mask when Encoder: src_mask
        ## mask Shape: [16, 90,]
        ## mask.unsqueeze(-1) Shape:[16, 90, 1]
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn
