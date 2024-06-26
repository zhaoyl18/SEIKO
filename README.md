<div align="center">

<!-- TITLE -->
# **Feedback Efficient Online Fine-Tuning of Diffusion Models**  

![SEIKO](assets/method.png)

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2402.16359-b31b1b.svg)](https://arxiv.org/abs/2402.16359)
</div>

This is the official implementation of paper [Feedback Efficient Online Fine-Tuning of Diffusion Models](https://arxiv.org/abs/2402.16359) accepted by [ICML 2024](https://openreview.net/forum?id=dtVlc9ybTm).

## Project Description

This study presents a novel reinforcement learning method to efficiently fine-tune diffusion models, targeting high-reward regions on the feasible manifold. The approach is validated both theoretically and empirically across images, biological sequences, and molecules. The repository includes the codebase for fine-tuning a pre-trained Stable Diffusion model in the image domain.

## Abstract

Diffusion models excel at modeling complex data distributions, including those of images, proteins, and small molecules. However, in many cases, our goal is to model parts of the distribution that maximize certain properties: for example, we may want to generate images with high aesthetic quality, or molecules with high bioactivity. It is natural to frame this as a reinforcement learning (RL) problem, in which the objective is to fine-tune a diffusion model to maximize a reward function that corresponds to some property. Even with access to online queries of the ground-truth reward function, efficiently discovering high-reward samples can be challenging: they might have a low probability in the initial distribution, and there might be many infeasible samples that do not even have a well-defined reward (e.g., unnatural images or physically impossible molecules). In this work, we propose a novel reinforcement learning procedure that efficiently explores on the manifold of feasible samples. We present a theoretical analysis providing a regret guarantee, as well as empirical validation across three domains: images, biological sequences, and molecules.

## Code

### Installation 

Create a conda environment with the following command:

```bash
conda create -n SEIKO python=3.10
conda activate SEIKO
pip install -r requirements.txt
```
Please use accelerate==0.17.0, other library dependancies might be flexible.

### Training

HuggingFace Accelerate will automatically handle parallel training.  
We conduct our experiments on image tasks using 4 A100 GPUs. Please adjust [*config.train.batch_size_per_gpu_available*] variable in config files according to your GPU memory.  

#### Running Non-adaptive (Baseline)  

```bash
accelerate launch online/online_main.py --config config/Non-adaptive.py:aesthetic
```

#### Running Greedy (Baseline)  

```bash
accelerate launch online/online_main.py --config config/Greedy.py:aesthetic
```

#### Running SEIKO-UCB  

```bash
accelerate launch online/online_main.py --config config/UCB.py:aesthetic
```

#### Running SEIKO-Bootstrap  

```bash
accelerate launch online/online_main.py --config config/Bootstrap.py:aesthetic
```

#### Running SEIKO-TS  

```bash
accelerate launch online/online_main.py --config config/TS.py:aesthetic
```

### Acknowledgement

Our codebase is directly built on top of [AlignProp](https://github.com/mihirp1998/AlignProp/) and [DDPO](https://github.com/kvablack/ddpo-pytorch).  
We are thankful to the authors for providing the codebases.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{
uehara2024feedback,
title={Feedback Efficient Online Fine-Tuning of Diffusion Models},
author={Masatoshi Uehara and Yulai Zhao and Kevin Black and Ehsan Hajiramezanali and Gabriele Scalia and Nathaniel Lee Diamant and Alex M Tseng and Sergey Levine and Tommaso Biancalani},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=dtVlc9ybTm}
}
```