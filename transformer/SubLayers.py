### Github[SubLayers.py]: https://github.com/ming024/FastSpeech2/blob/master/transformer/SubLayers.py
## Github[utils.tools]: https://github.com/ming024/FastSpeech2/blob/master/utils/tools.py#L265

### Github[train.py] https://github.com/ming024/FastSpeech2/blob/master/train.py
### Forward
### output = model(*(batch[2:])) 

### Github[model.fastspeech2.py] : https://github.com/ming024/FastSpeech2/blob/master/model/fastspeech2.py
### output = self.encoder(texts, src_masks)
### texts is the only input of the Encoder

### Github[transformer.Constants.py]: https://github.com/ming024/FastSpeech2/blob/master/transformer/Constants.py
# PAD = 0
# UNK = 1
# BOS = 2
# EOS = 3

# PAD_WORD = "<blank>"
# UNK_WORD = "<unk>"
# BOS_WORD = "<s>"
# EOS_WORD = "</s>"

#### Github[config.LibriTTS.model.yaml]: https://github.com/ming024/FastSpeech2/blob/master/config/LibriTTS/model.yaml
# transformer:
#   encoder_layer: 4
#   encoder_head: 2
#   encoder_hidden: 256
#   decoder_layer: 6
#   decoder_head: 2
#   decoder_hidden: 256
#   conv_filter_size: 1024
#   conv_kernel_size: [9, 1]
#   encoder_dropout: 0.2
#   decoder_dropout: 0.2

import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head = 2, d_model=256, d_k = None, d_v =  None, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.head_dim = d_model // n_head
        self.n_head = n_head
        # self.d_k = d_k
        # self.d_v = d_v
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.linear_dim = (d_model, d_model)
        
        ## This NEW
        self.w_qs, self.w_ks, self.w_vs, self.fc = [copy.deepcopy( nn.Linear(*self.linear_dim) ) for _ in range(4)]
        self.linears = nn.ModuleList([self.w_qs, self.w_ks, self.w_vs, self.fc])

        ## This is Before
        # self.linears = nn.ModuleList([copy.deepcopy( nn.Linear(*self.linear_dim) ) for _ in range(4)])
        # self.w_qs, self.w_ks, self.w_vs, self.fc = self.linears  

        ## Original
        # self.fc = nn.Linear(*self.linear_dim)
        # self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature = np.power(self.head_dim, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        residual = q
        bs, len_input, _ = q.size()

        # q, k, v = [l(x).view(bs, -1, n_head, head_dim).transpose(1, 2). for l, x in zip(linears, (q, k, v))]
        q, k, v = [l(x).view(bs, -1, self.n_head, self.head_dim).permute(2, 0, 1, 3).contiguous().view(-1, len_input, self.head_dim) for l, x in zip(self.linears, (q, k, v))]

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        ## output: [bs * n_head, len_input, head_dim] -> [ bs, n_head, len_input, head_dim]
        ##          -> [n_head, bs, len_input, head_dim] -> [bs, len_input, ]
        output = output.view(self.n_head, -1, len_input, self.head_dim).permute(1, 2, 0, 3).contiguous().view(bs, len_input, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        ## d_in = 256 ## encoder_hidden or d_model in MHA
        ## d_hid = 1024

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0], ## kernel_size = [9, 1]
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ### x(=input): x = mha_output1.masked_fill(src_masks.unsqueeze(-1), 0) ## [16, 90, 256]
        ### mha_output1: from MultiHeadAttentio n## [16, 90, 256]

        residual = x
        output = x.transpose(1, 2) ## [16, 90, 256] -> [16, 256, 90]
        output = self.w_2(F.relu(self.w_1(output))) ## [16, 256, 90] -> [16, 1024, 90] -> [16, 256, 90]
        output = output.transpose(1, 2) ## [16, 256, 90] -> [16, 90, 256]
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


