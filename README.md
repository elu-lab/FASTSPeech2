# FASTSPeech2
 TTS(= Text-To-Speech) Model for researching. This Repository is mainly based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and modified or added some codes for our team's dataset. We use [MLS(=Multilingual LibriSpeech)](https://www.openslr.org/94/) dataset for training. 


## wandb :waning_gibbous_moon:
 If you wanna see the training status, you can check here [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FastSpeech2_german)
You can check theses things above wandb link:
- Listen to the Samples(= Label Speech & predicted Speech)
- Training / Eval's Mel-Spectrogram


## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [🤗 Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [🤗 Accelerate(Github)](https://github.com/huggingface/accelerate) 
- [🤗 examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
