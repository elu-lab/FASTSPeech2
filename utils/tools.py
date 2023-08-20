
## Github [utils.tools]: https://github.com/ming024/FastSpeech2/blob/master/utils/tools.py
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################# @ Length Regulator #################################
def pad(input_ele, mel_max_length=None):
    # input_ele : output from 'expand' # [16, 1465, 256]
    # mel_max_length : # max_mel_len from Dataloader # equal to max_len in LR
    if mel_max_length:
        max_len = mel_max_length # 1465
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad( batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            ## batch: torch.Size([598, 256])
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
            ## torch.Size([1465, 256]))
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


############################# @ transformer #################################
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


#################### @ dataset.py - Dataset ################
def pad_1D(inputs, PAD=0):

    ### function in function ###
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    ######## pad_1D ###########
    max_len = max((len(x) for x in inputs)) ### 가지고 있는 것 중 제일 긴 거?
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):
    
    ### function in function ###
    def pad(x, max_len):
        PAD = 0
        ##### ROW #####
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]
        
    ######## pad_2D ###########
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        ## 없으면 가지고 있는 것 중 제일 긴 거?
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


##################### @dataloader-test and others #######################
def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)
