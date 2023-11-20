
import os
import yaml
import gc 

import re
from string import punctuation

from requests import post
from phonepiece.ipa import read_ipa        

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
# from IPython.display import Audio
from matplotlib.patches import Rectangle

import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path
from PIL import Image

from utils.tools import * ## NOT GPU
from utils.model import * ## NOT GPU

from accelerate import Accelerator

from torchmalloc import *
from evaluate import *

from dataset import TextDataset
from text import text_to_sequence

from requests import post
from phonepiece.ipa import read_ipa      


def g2p_call(lang = 'ko', text = ["안녕하세요."], url: str = 'http://a6000.elulab.kr:50000'):

    response = post(f'{url}/{lang}', json={'text': text})
    if response.status_code != 200:
        raise SystemError('Did you run the "mfa_g2p_service.py" before run this file?')
    
    return response.json()['phon']
    # board.elulab.kr:50000 
    # a6000.elulab.kr:50000 --> Seems working

    # def g2p_call(lang = 'de', text = [sen], url: str = 'http://localhost:5000'):
    # def g2p_call(lang = 'de', text = [sen], url: str = 'http://4090.elulab.kr:5000'):


def preprocess_g2p(lang, text):
    # from phonepiece.ipa import read_ipa                                                         
    ipa = read_ipa()
    outs = g2p_call(lang, [text])
    temp = ipa.tokenize(outs[0])  
    phones = "{" + "}{".join(temp) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    # print(phones)

    # sequence = np.array(text_to_sequence( phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]))
    sequence = np.array(text_to_sequence( phones, ['transliteration_cleaners']))

    return sequence, phones


def convert_to_inputs(raw_texts, lang = 'ko'):

    ### 1) Speaker_id: Single Speaker 
    speakers = np.array([8505]) 

    ### 2) Professor's G2P
    sequence, phones = preprocess_g2p(lang, raw_texts)
    # print("Sequence: ",sequence)
    print(f"SENTENCE: {raw_texts}")
    print("Phones: ", phones)
    texts = sequence.reshape(1, -1)
    text_lens = np.array([len(texts[0])])

    ids = 'sytnh'
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
    return batchs, phones


@torch.inference_mode()
def synthesize_fn(model, configs, batchs, control_values, device, vocoder, vocoder_train_setup=None, denoiser = None, denoising_strength=0.005):
    
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    model.eval()
    with torch.no_grad():
        for batch in batchs:
            cuda_batch = to_device(batch, device)
            # Forward
            output = model(*(cuda_batch[2:]), 
                            p_control=pitch_control, 
                            e_control=energy_control, 
                            d_control=duration_control)
            
            # Synthesize
            synth_samples(cuda_batch, output, model_config, preprocess_config, train_config["path"]["result_path"],
                        vocoder, vocoder_train_setup, denoiser, denoising_strength)


def syn(raw_texts, 
        model, 
        device,
        configs,
        lang = 'ko', 
        control_values = (1.0, 1.0, 1.0), 
        accelerator = None):
    
    preprocess_config, model_config, train_config = configs

    # Load vocoder
    vocoder, vocoder_train_setup, denoiser = get_vocoder(model_config, device)
    # control_values = args.pitch_control, args.energy_control, args.duration_control
    control_values = control_values 

    if accelerator is not None:
        ## NVIDIA-HiFi-GAN
        model, vocoder, denoiser = accelerator.prepare(model, vocoder, denoiser)
    else:
        model = model.to(device)
        vocoder = vocoder.to(device)
        denoiser = denoiser.to(device)

    batchs, phones = convert_to_inputs(raw_texts, lang = lang)
    ids = batchs[0][0]
    synthesize_fn(model, configs, batchs, control_values, accelerator.device, vocoder, vocoder_train_setup, denoiser, 0.005)
    print("Synthesize Completed")

    ## AUDIO, MEL SAVE PATH
    audio_result_path = train_config["path"]["result_path"] + f"/{ids[0]}.wav"
    mel_result_path = train_config["path"]["result_path"] + f"/{ids[0]}.png"
    print(f"SAVED PATHS: AUDIO @ {audio_result_path}")
    print(f"SAVED PATHS: Mel_Spectrogram @ {audio_result_path}")

    return ids, raw_texts, phones, audio_result_path , mel_result_path 



def main(args, configs):

    (preprocess_config, model_config, train_config) = configs

    # from accelerate import Accelerator
    accelerator = Accelerator()

    device = accelerator.device
    print(device)

    model_id = args.restore_step # default: 53100
    model = get_model(args, configs, device = device, train=False)
    print(f"{model_id}Model Loaded", end ="\n")


    raw_texts = args.raw_texts
    ids, raw_texts, phones, audio_result_path, mel_result_path = syn(raw_texts, 
                                                                     model = model, 
                                                                     device = accelerator.device,
                                                                     configs=configs,
                                                                     lang = 'ko', 
                                                                     control_values = (1.0, 1.0, 1.0 ), 
                                                                     accelerator=accelerator)
    # print("Synthesize Completed")
    # print(f"MODEL ID: {model_id}")
    # print(f"SENTENCE: {raw_texts}")
    # print(f"Pure Length {len(raw_texts)}")
    # print(f"Phones: {phones}")
    # print(audio_result_path)
    # print(mel_result_path)
    # Image.open(mel_result_path).convert("RGB")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_texts', type = str, default = "안녕하세요. 이엘유 연구실입니다.", help="Text")
    parser.add_argument("--restore_step", type=int, default=53100)

    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        # required=True,
        default = "./config/LibriTTS/preprocess.yaml",
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", 
        "--model_config", 
        type=str, 
        # required=True, 
        default = "./config/LibriTTS/model.yaml",
        help="path to model.yaml"
    )
    parser.add_argument(
        "-t", 
        "--train_config", 
        type=str, 
        # required=True, 
        default = "./config/LibriTTS/train.yaml",
        help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    # preprocess_config = yaml.load(open("./config/LibriTTS/preprocess.yaml", "r"), Loader=yaml.FullLoader)
    # train_config = yaml.load(open("./config/LibriTTS/train.yaml", "r"), Loader=yaml.FullLoader)
    # model_config = yaml.load(open("./config/LibriTTS/model.yaml", "r"), Loader=yaml.FullLoader)

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
