# FASTSPeech2
 TTS(= Text-To-Speech) Model for researching. This Repository is mainly based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and modified or added some codes for our team's dataset. We use [MLS(=Multilingual LibriSpeech)](https://www.openslr.org/94/) dataset for training. 


## Languages
 We trained FastSpeech2 Model following languages.
- German
- English


## wandb [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FastSpeech2_german)
 If you wanna see the training status, you can check here. You can check theses things above [`wandb link`](https://wandb.ai/wako/FastSpeech2_german):
- Listen to the Samples(= Label Speech & predicted Speech)
- Training / Eval's Mel-Spectrogram

## Train
 First, you should log-in wandb with your token key in CLI. 
```
wandb login --relogin '##### Token Key #######'
```

 Next, you can set your training environment with following commands. 
```
accelerate config
```

 With this command, you can start training. 
```
accelerate launch train.py --n_epochs 990 --save_epochs 50 --synthesis_logging_epochs 30 --try_name T4_MoRrgetda
```

Also, you can train your TTS model with this command.
```
CUDA_VISIBLE_DEVICES=0,3 accelerate launch train.py --n_epochs 990 --save_epochs 50 --synthesis_logging_epochs 30 --try_name T4_MoRrgetda
```

 Inference


 Synthesize


## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [pytorch_hub/nvidia/HIFI-GAN](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/)
- [🤗 Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [🤗 Accelerate(Github)](https://github.com/huggingface/accelerate) 
- [🤗 examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
