<div align="center">
  <a ><img width="300px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/LOGO%20sparrow.svg"></a>
</div>

---

**Sparrow** is a Reinforcement Learning Friendly Simulator for Mobile Robot

## Features:

- **Vectorizable** (Fast data collection; Also support single Environment)
- **Domain Randomization** (maps, inertia, friction, sensor noise, control period, control delay, ...)
- **Lightweight** (30kb, pure Python files. Only import, never worry about installation)
- **Accept image as maps** (Customize your own environments easily and rapidly)
- **Ubuntu/Windows** are both supported
- **Standard Gym API with both Pytorch/Numpy data flow**
- **GPU/CPU are both acceptable** (If you use Pytorch to build your RL model, you can run you RL model and Sparrow both on GPU. Then you don't need to transfer the transitions from CPU to GPU anymore.)

## Installation:

The dependencies for Sparrow are:

```bash
gym >= 0.26.2
pygame >= 2.1.2
numpy >= 1.24.2
torch >= 1.13.1
```

You can install **gym**, **pygame**, **numpy** via:

```bash
pip3 install gym==0.26.2 pygame==2.1.2 numpy==1.24.2
```

You can install **torch** by following the guidence from its [Official website](https://pytorch.org/get-started/locally/). We strongly suggest you install the **CUDA 11.7** version.

## Quick Start:

You can play with Sparrow with your keyboard (up/down/left/right botton) via:

```bash
python play_with_keyboard.py
```

## Train a DQN agent with Sparrow:

```bash
python train_DQN.py
```
