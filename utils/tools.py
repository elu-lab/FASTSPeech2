
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

