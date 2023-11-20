# FASTSPeech2
 TTS(= Text-To-Speech) Model for researching. This Repository is mainly based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) and modified or added some codes for our team's dataset. We use [AI-HUB: Multi-Speaker-Speech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=542) dataset and [MLS(=Multilingual LibriSpeech)](https://www.openslr.org/94/) dataset for training. 


## Languages
 We trained FastSpeech2 Model following languages with introducing each language's phonsets we embedded and trained. We used [`Montreal-Forced Alignment`](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) tool to obtain the alignments between the utterances and the phoneme sequences as described in the [paper](https://arxiv.org/pdf/2006.04558.pdf). As you can see, we embedded [`IPA Phoneset`](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet). 
<details>
<summary>Korean</summary>
<div>
<a href="https://mfa-models.readthedocs.io/en/latest/dictionary/Korean/Korean%20MFA%20dictionary%20v2_0_0a.html">Korean MFA dictionary v2.0.0a</a> : <code style="white-space:nowrap;"> 'b d dʑ e eː h i iː j k kʰ k̚ k͈ m n o oː p pʰ p̚ p͈ s sʰ s͈ t tɕ tɕʰ tɕ͈ tʰ t̚ t͈ u uː w x ç ŋ ɐ ɕʰ ɕ͈ ɛ ɛː ɡ ɣ ɥ ɦ ɨ ɨː ɭ ɰ ɲ ɸ ɾ ʌ ʌː ʎ ʝ β'</code>
</div>
</details>

<details>
<summary>German</summary>
<div>
<a href="https://mfa-models.readthedocs.io/en/latest/dictionary/German/German%20MFA%20dictionary%20v2_0_0a.html">German MFA dictionary v2.0.0a</a> : <code style="white-space:nowrap;">a aj aw aː b c cʰ d eː f h iː j k kʰ l l̩ m m̩ n n̩ oː p pf pʰ s t ts tʃ tʰ uː v x yː z ç øː ŋ œ ɐ ɔ ɔʏ ə ɛ ɟ ɡ ɪ ɲ ʁ ʃ ʊ ʏ</code>
</div>
</details>

<details>
<summary>English(US)</summary>
<div>
<a href="https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20MFA%20dictionary%20v2_2_1.html">English MFA dictionary v2.2.1</a> : <code style="white-space:nowrap;">a aj aw aː b bʲ c cʰ cʷ d dʒ dʲ d̪ e ej f fʲ fʷ h i iː j k kp kʰ kʷ l m mʲ m̩ n n̩ o ow p pʰ pʲ pʷ s t tʃ tʰ tʲ tʷ t̪ u uː v vʲ vʷ w z æ ç ð ŋ ɐ ɑ ɑː ɒ ɒː ɔ ɔj ə əw ɚ ɛ ɛː ɜ ɜː ɝ ɟ ɟʷ ɡ ɡb ɡʷ ɪ ɫ ɫ̩ ɲ ɹ ɾ ɾʲ ɾ̃ ʃ ʉ ʉː ʊ ʎ ʒ ʔ θ</code>
</div>
</details>



## wandb [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FastSpeech2_german)
 If you wanna see the training status, you can check here. You can check theses things above [`wandb link`](https://wandb.ai/wako/FastSpeech2_german):
- Listen to the Samples(= Label Speech & predicted Speech)
- Training / Eval's Mel-Spectrogram

## Dataset
- [AI-HUB: Multi-Speaker-Speech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=542)
- [MLS(=Multilingual LibriSpeech)](https://www.openslr.org/94/) 


## Languages
 We trained FastSpeech2 Model following languages with introducing each language's phonsets we embedded and trained. We used [`Montreal-Forced Alignment`](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) 



## Preprocess
 This `preprocess.py` can give you the pitch, energy, duration and phones from `TextGrid` files. 
```
python preprocess.py config/LibriTTS/preprocess.yaml 
```

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

## Synthesize
you can synthesize speech in CLI with this command: 
```
python synthesize.py --raw_texts <Text to syntheize to speech> --restore_step 53100
```
Also, you can check this [jupyter-notebook](https://github.com/elu-lab/FASTSPeech2/blob/main/synthesize_example.ipynb) when you try to synthesize.



## References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [HGU-DLLAB/Korean-FastSpeech2-Pytorch
Public](https://github.com/HGU-DLLAB/Korean-FastSpeech2-Pytorch)
- [pytorch_hub/nvidia/HIFI-GAN](https://pytorch.org/hub/nvidia_deeplearningexamples_hifigan/)
- [🤗 Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator)
- [🤗 Accelerate(Github)](https://github.com/huggingface/accelerate) 
- [🤗 huggingface/peft/.../peft_lora_clm_accelerate_ds_zero3_offload.py](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_accelerate_ds_zero3_offload.py)
