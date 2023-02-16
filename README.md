<div align="center">
  <a ><img width="300px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/LOGO%20sparrow.svg"></a>
</div>

# A Reinforcement Learning Friendly Simulator for Mobile Robot


![Python](https://img.shields.io/badge/Python-blue)
![DRL](https://img.shields.io/badge/DRL-blueviolet)
![Mobile Robot](https://img.shields.io/badge/MobileRobot-ff69b4)


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

After installation, you can play with Sparrow with your keyboard (up/down/left/right botton):

```bash
python play_with_keyboard.py
```

## Train a DQN agent with Sparrow:

```bash
python train_DQN_single.py
```

```bash
python train_DQN_vector.py
```

## Dive into Sparrow:

### Create your first env

Sparrow has standard [Gym](https://www.gymlibrary.dev/) API. Thus, you can create the Sparrow environment via:

```python
import SparrowV0
import gym

env = gym.make('Sparrow-v0')
```

The above commond will create a standard Gym environment, and you can iteract with it via:

```python
s, info = env.reset()
done = False
While not done:
    a = model(s)
    s_next, r, terminated, truncated, info = env.step(a)
    done = terminated + truncated
```

As we have mentioned, Sparrow is **vectoriable**, so you can create a vectorized environment via:

```python
import SparrowV0
import gym

envs = gym.vector.AsyncVectorEnv([lambda: gym.make('Sparrow-v0', np_state = True)  for _ in range(N)])
```

Here, `N`is the number of vectorized environments. In this context, the RL model should iteract with the environment in a batched manner. And the dimension of **s, a, r, terminated, truncated** are **(N,32), (N,), (N,), (N,), (N,)** respectively. **np_state=Ture** means that the state will be returned in *numpy.narray*. More parameter setting will be introduced in the next section.

### Basic parameters

There are 9 parameters you could configure when creating Sparrow:

```python
env = gym.make('Sparrow-v0',dvc, ld_num, np_state, colorful, state_noise, render_mode, render_speed, evaluator_mode, eval_map)
```

- **dvc (string; default `'cuda:0'`)**:
  - The device that runs the Sparrow
    
  - Should be one of `'cpu'`/`'cuda:0'`. We suggest using `'cuda:0'` (GPU) to accelerate simulation
    
- **ld_num (int; default `27`)**: number of LiDAR rays.
- **np_state (bool; default `False`)**:
  - `False`: return state in *torch.tensor* (will be on same device as dvc)
    
  - `True`: return state in *numpy.ndarray* (will be on CPU)
    
  - When using vectorized Env, it should be True, because gym.vector only support data in *numpy.ndarray*
    
- **colorful (bool; default `False`)**: if `True`, the follow items will be randomized:
  - physical parameters (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, magnitude of noise)
    
  - initial position of the robot at the beginning of each episode
    
  - maps
    
- **state_noise (bool; default `False`)**: if `True`, the state of the robot will contain uniformly distributed noise (it doesn't impact the accuracy of simulation)
- **render_mode (string; default `None`)**: should be one of `"human"` /`"rgb_array"`/ `None`
  - `"human"`: render in a pygame window
    
  - `"rgb_array"`: *env.render()* will return the airscape of the robot in *numpy.ndarray*
    
  - `None`: not render anything
    
- **render_speed (string; default `'fast'`)**: control the rendering speed, should be one of `'fast'`/`'real'`/`'slow'`
- **evaluator_mode (bool; default `False`)**: if `True`, *env.reset()* will not swap maps and robot will always be initialized in bottom left corner.
- **eval_map (string; default `None`)**: if *evaluator_mode=True*, you need to designate the map on which you want to evaluate. And *eval_map* should be its absolute address, e.g. `os.getcwd()+'SparrowV0/envs/train_maps/map4.png'`

