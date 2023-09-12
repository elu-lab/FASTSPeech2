########## Github[utils.model.py]: https://github.com/ming024/FastSpeech2/blob/master/utils/model.py ################

import os
import json

import torch
import numpy as np

# import hifigan
# from model import FastSpeech2, ScheduledOptim

# from model.modules import *
from model.fastspeech2 import *
# from model.loss import *
from model.optimizer import *


################################# @ train.py #####################################
def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config, device).to(device)
    if args is not None and args.restore_step:
        ckpt_path = os.path.join(train_config["path"]["ckpt_path"], "model_{}.pth".format(args.restore_step),)
        ckpt = torch.load(ckpt_path, map_location=device)
        # model.load_state_dict(ckpt["model"])
        model.load_state_dict(ckpt)

    if train:
        scheduled_optim = ScheduledOptim( model, train_config, model_config, 0)
        if args is not None and args.restore_step:
            scheduled_optim = ScheduledOptim( model, train_config, model_config, args.restore_step)
            ckpt_opim_path = torch.load(os.path.join(train_config["path"]["ckpt_path"], "optimizer_{}.pth".format(args.restore_step),), map_location=device)
            # scheduled_optim.load_state_dict(ckpt["optimizer"])
            scheduled_optim.load_state_dict(ckpt_opim_path)
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


################################# Original #####################################
# def get_model(args, configs, device, train=False):
#     (preprocess_config, model_config, train_config) = configs

#     model = FastSpeech2(preprocess_config, model_config).to(device)
#     if args.restore_step:
#         ckpt_path = os.path.join(
#             train_config["path"]["ckpt_path"],
#             "{}.pth.tar".format(args.restore_step),
#         )
#         ckpt = torch.load(ckpt_path)
#         model.load_state_dict(ckpt["model"])

#     if train:
#         scheduled_optim = ScheduledOptim(
#             model, train_config, model_config, args.restore_step
#         )
#         if args.restore_step:
#             scheduled_optim.load_state_dict(ckpt["optimizer"])
#         model.train()
#         return model, scheduled_optim

#     model.eval()
#     model.requires_grad_ = False
#     return model



################################# @ train.py #####################################
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


###################### Test:nvidia/HiFiGAN #########################
def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "linda_johnson" )
        elif speaker == "universal":
            vocoder = torch.hub.load( "descriptinc/melgan-neurips", "load_melgan", "multi_speaker" ) ## Doesn't work
            ### mel() takes 0 positional arguments but 5 were given
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
        return vocoder
    elif name == "HiFi-GAN":
        # with open("hifigan/config.json", "r") as f:
        #     config = json.load(f)
        # config = hifigan.AttrDict(config)
        # vocoder = hifigan.Generator(config)
        # if speaker == "LJSpeech":
        #     ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        # elif speaker == "universal":
        #     ckpt = torch.load("hifigan/generator_universal.pth.tar")
        # vocoder.load_state_dict(ckpt["generator"])

        # hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan') ## Worked
        vocoder, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan') ## Worked
        
        vocoder.eval()
        # vocoder.remove_weight_norm()
        vocoder.to(device)

        return vocoder, vocoder_train_setup, denoiser 


###################### Original ##################################
# def get_vocoder(config, device):
#     name = config["vocoder"]["model"]
#     speaker = config["vocoder"]["speaker"]

#     if name == "MelGAN":
#         if speaker == "LJSpeech":
#             vocoder = torch.hub.load(
#                 "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
#             )
#         elif speaker == "universal":
#             vocoder = torch.hub.load(
#                 "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
#             )
#         vocoder.mel2wav.eval()
#         vocoder.mel2wav.to(device)
#     elif name == "HiFi-GAN":
#         with open("hifigan/config.json", "r") as f:
#             config = json.load(f)
#         config = hifigan.AttrDict(config)
#         vocoder = hifigan.Generator(config)
#         if speaker == "LJSpeech":
#             ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
#         elif speaker == "universal":
#             ckpt = torch.load("hifigan/generator_universal.pth.tar")
#         vocoder.load_state_dict(ckpt["generator"])
#         vocoder.eval()
#         vocoder.remove_weight_norm()
#         vocoder.to(device)

#     return vocoder


################################# @ train.py - HiFiGAN #####################################
def vocoder_infer(mels, model_config, preprocess_config, 
                  vocoder, vocoder_train_setup = None, denoiser = None, denoising_strength = 0.005,
                  lengths=None):
    
    name = model_config["vocoder"]["model"]
    
    with torch.no_grad():
    
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))

        elif name == "HiFi-GAN":
            ##### HifiGAN Vocoder  Colab
            # mel, mel_lens, *_ = fastpitch(batch['text'].to(device), **gen_kw)
            # audios = hifigan(mel).float()
            # audios = denoiser(audios.squeeze(1), denoising_strength)
            # audios = audios.squeeze(1) * vocoder_train_setup['max_wav_value']

            ### Original
            # wavs = vocoder(mels).squeeze(1)
            wavs = vocoder(mels).float()
            wavs = denoiser(wavs.squeeze(1), denoising_strength)
            wavs = wavs.squeeze(1) # * vocoder_train_setup['max_wav_value']


    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
