# [NeurIPS 2025 Spotlight] LaViDa:A Large Diffusion Language Model for Multimodal Understanding

[[Paper]](paper/paper.pdf) [[Arxiv]](https://arxiv.org/abs/2505.16839) [[Checkpoints]](https://huggingface.co/collections/jacklishufan/lavida-10-682ecf5a5fa8c5df85c61ded) [[Data]](https://huggingface.co/datasets/jacklishufan/lavida-train) [[Website]](https://homepage.jackli.org/projects/lavida/)

## News

**Dec 16 2025** We released the [preprint](https://arxiv.org/abs/2512.14008) and  [Project Page](https://homepage.jackli.org/projects/sparse-lavida/index.html) for **Sparse-LaViDa**, an efficient optimization technique for training and sampling from unified multi-modal dLLMs based on LaViDa. 

**Oct 2025:** We opensourced [LaViDa-O](https://github.com/adobe-research/LaVida-O), a state-of-the-art unified multi-modal built on LaViDa

**Sep 2025**  We released the preprint for [LaViDa-O](https://arxiv.org/abs/2509.19244), an extension of LaViDa to visual generation tasks.

**Aug 2025:** Our work was accepted to NeurIPS 2025 as a Spotlight Paper!


![Model Architecture](images/architecture.png)
![Speedup](images/demo_hd.gif)

### Installation

```
conda create --name lavida python=3.13
conda activate lavida
pip install -e .[train]
cd eval
pip install -e .
cd ../
pip install trl==0.17.0 
```

### Transformers-Compatible Checkpoint
For easy reproducibility, inference, and testing, we provide a Transformers-compatible checkpoint that does not require the source code to run.
Please download checkpoints from [Huggingface](huggingface.co/KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers)

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('./lavida-llada-v1.0-instruct/')
model = AutoModelForCausalLM.from_pretrained('./lavida-llada-v1.0-instruct/', torch_dtype=torch.bfloat16)
image_processor = model.get_vision_tower().image_processor

model.resize_token_embeddings(len(tokenizer))
model.tie_weights()
```

### Download Checkpoint

Please download checkpoints from [Huggingface](https://huggingface.co/collections/jacklishufan/lavida-10-682ecf5a5fa8c5df85c61ded)  and organize them in the following structure
```
<repo root>
--lavida-ckpts # create this folder via mkdir
   --lavida-llada-hd # jacklishufan/lavida-llada-v1.0-instruct
   --lavida-dream-hd # jacklishufan/lavida-dream-v1.0-instruct
   --lavida-llada-hd-fim  # jacklishufan/lavida-llada-1.0-fim
   --lavida-llada-hd-reason # hbXNov/lavida-llada-reason
   --lavida-llada-lowres  # jacklishufan/lavida-llada-1.0-lowres
```

## Inference
run the following script to perfom standard inference and text-infilling
```
python predict.py
python predict_fim.py
```
## Evaluation
### Reproduce Main Evaluation Results

| Model         | MME    | MMMU | MMB  | Latency (s/image) |
|---------------|--------|------|------|-------------------|
| LaViDa-Dream  | **1463.5** | 42.6 | **73.8** | **1.13**              |
| LaViDa-LLaDa  | 1365.6 | **43.3** | 70.5 | 1.32              |
| [MMaDa](https://github.com/Gen-Verse/MMaDA)         | 1410.7 | 30.2 | 68.5 | 3.93             |

(speed measurement conducted with generation length=32 and steps=16)

The evaluation scrips are under `eval` folder. Please use the following script to reproduce the main results on MMMU.

```
bash eval/run.sh lavida-ckpts/lavida-llada-hd --tasks mmmu_val # for LaViDa-LLaDa
bash eval/run_dream.sh lavida-ckpts/lavida-dream-hd --tasks mmmu_val # for LaViDa-Dream
```


To reproduce results on other dataset, simply replace `mmmu_val` to respective dataset.

### Reproduce COCO Caption Results (Speed-Quality Tradeoff)


```
bash eval/run_coco.sh lavida-ckpts/lavida-llada-hd 
```

| Model   | KV Cache | CIDEr $\uparrow$ | Latency $\downarrow$  | NFE  |
|------------|----------|-------|---------|-----|
| LaviDa-LLaDa  | off      | 110.2 | 6.65    | 100\%  |
| LaviDa-LLaDa  | on      | 107.8 |  2.01   | 100\%  |
| LaviDa-LLaDa  | off      | 108.5 | 3.57    | 50\%  |
| LaviDa-LLaDa  | on       | 104.4 | 1.32    | 50\%  |
| LLaVa-1.6-7B (Baseline)  | on       |  96.7 | 1.67     | 100\%|

We find that the low resolution model is slightly faster than HD model and have stronger performance on some tasks (e.g. COCO captioning). We provide the inference script as well.

```
bash eval/run_coco_lowres.sh lavida-ckpts/lavida-llada-lowres 
```


## Training
### Data Preparation

The expected data folder structure looks like the following

```
<repo root>
--data
   --pretrain # LCS-558K
      -- images
      -- blip_laion_cc_sbu_558k.json
   --Open-LLaVA-NeXT
      -- ai2d
      -- ...
      -- open-llava-next 
   --infovqa-v1
   --VQAv2_train
```


1. Download [LCS-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) and place it in `data/pretrain`
2. Download all datasets from [Open-LLaVa-Next](https://github.com/xiaoachen98/Open-LLaVA-NeXT/blob/master/docs/Data.md) and place it in `data/Open-LLaVa-Next`
3. Download the remaining datasets from [Our Huggingface](https://huggingface.co/datasets/jacklishufan/lavida-train/tree/main). This dataset contains three subfolders

```
infovqa-v1 -> put under data/
VQAv2_train -> put under data/
open-llava-next -> put under data/Open-LLaVA-NeXT/, merge with an existing folder of same name
```

### Launch Scripts

```
Pretrain(Stage 1) Scripts:
scripts/train/exps/cluster/pretrain_llada.sh
scripts/train/exps/cluster/pretrain_dream.sh

Finetune(Stage 2) Scripts

scripts/train/exps/cluster/llada-hd-llada-s2.sh
scripts/train/exps/cluster/llada-hd-dream-s2.sh
```

To launch finetuning scripts, you need to change the `BASE_RUN_NAME` variable in the shell scripts to the path of stage 1 checkpoints. If you want to directly launch stage 2 training, we provide pretrained stage 1 checkpoints in the link [Stage-1-LLaDa](https://huggingface.co/jacklishufan/lavida-llada-1.0-stage1) and [Stage-1-Dream](https://huggingface.co/jacklishufan/lavida-dream-1.0-stage1)

## Common Questions

### Why validation acc is 0 during the training

We observed a bug with deepspeed Zero-3 that it breaks inference for validation. Hence, if you want to start a training run with eval results logged to wandb, please use Zero-2.

### Where is Reasoning Data used in Stage 3

It can be found in the huggingface [collection](https://huggingface.co/collections/jacklishufan/lavida-10-682ecf5a5fa8c5df85c61ded)

### How to launch FIM training

The scripts is in `scripts/train/exps/cluster/llada-hd-llada-s3-fim.sh`

### Acknowledgements

This repo is largely based on [LLaVa-Next](https://github.com/LLaVA-VL/LLaVA-NeXT). We use [LMMS-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluations. 

### Citation

```
@inproceedings{lilavida,
  title={LaViDa: A Large Diffusion Model for Vision-Language Understanding},
  author={Li, Shufan and Kallidromitis, Konstantinos and Bansal, Hritik and Gokul, Akash and Kato, Yusuke and Kozuka, Kazuki and Kuen, Jason and Lin, Zhe and Chang, Kai-Wei and Grover, Aditya},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
@article{li2025lavida,
  title={Lavida-O: Elastic Large Masked Diffusion Models for Unified Multimodal Understanding and Generation},
  author={Li, Shufan and Gu, Jiuxiang and Liu, Kangning and Lin, Zhe and Wei, Zijun and Grover, Aditya and Kuen, Jason},
  journal={arXiv preprint arXiv:2509.19244},
  year={2025}
}
```
