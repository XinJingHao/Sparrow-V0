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

After installation, you can play with Sparrow with your keyboard (up/down/left/right button) via:

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

Since Sparrow has standard [Gym](https://www.gymlibrary.dev/) API, you can create the Sparrow environment via:

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
    a = model(s) # your own RL model
    s_next, r, terminated, truncated, info = env.step(a)
    done = terminated + truncated
```

As we have mentioned, Sparrow is **vectoriable**, so you can create a vectorized environment via:

```python
import SparrowV0
import gym

envs = gym.vector.AsyncVectorEnv([lambda: gym.make('Sparrow-v0', np_state = True) for _ in range(N)])
```

Here, `N`is the number of vectorized environments. In this context, the RL model should iteract with the environment in a batched manner. And the dimension of **s, a, r, terminated, truncated** are **(N,32), (N,), (N,), (N,), (N,)** respectively. Note that **np_state=Ture** means the state will be returned in *numpy.narray*. More parameter setting will be introduced in the next section.

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
    
  - When using vectorized Env, np_state should be True, because gym.vector only support data in *numpy.ndarray*
    
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
  

### Coordinate Frames

There are three coordinate frames in Sparrow as shows bellow. The ground truth position of the robot is calculated in **World Coordiante Frame**, which will be normalized and represented in **Relative Coordiante Frame** to comprise the RL state variable. The **GridCoordiante Frame** comes from *pygame*, used to draw the robot, obstacles, target area, etc.

<div align="center">
<img width="360px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/coordinate_frames.svg">
</div>


### Basic Robot Information

The LiDAR perception range is 100cm×270°, with accuracy of 3 cm. The radius of the robot is 9 cm, and its collision threshold is 14 cm. The maximum linear and angular velocity of the robot is 18 cm/s and 1 rad/s, respectively. The control frequency of the robot is 10Hz. 

<div align="center">
<img width="360px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/basic_robot_info.svg">
</div>

We use a simple but useful model to discribe the kinematics of the robot

$$[V^{t+1}_{linear},\ V^{t+1}_{angular}] = K·[V^{t}_{linear},\ V^{t}_{angular}]+(1-K)·[V^{target}_{linear},\ V^{target}_{angular}]$$

Here, **K** is a hyperparameter between (0,1), discribing the combined effect of inertia and friction, default: 0.6. The parameters mentioned in this section can be found in the *Robot initialization* and *Lidar initialization* part of `SparrowV0/envs/sparrow_v0.py` and customized according to your own scenario.

### RL representation

The basic task in Sparrow is about driving the robot from the start point to the end point without colliding with the obstacles as fast as possible. To this end, in the following sub sections, we will define several basic components in Markove Decision Process.

#### State:

The state of the robot is a vector of lenth 32, containning **position** (*state[0:2] = [dx,dy]*), **orientation** (*state[2]=α*), **Velocity** (*state[3:5]=[v_linear, v_angular]*), **LiDAR** (*state[5:32] = scanning result*). The **position** and **orientation** are illustrated as follows:

<div align="center">
<img width="360px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/state_train.svg">
</div>

These state variables will be normalized into the Relative Coordiante Frame before outputed to the RL model. The position is normalized through:

$$[dx_{rt},\ dy_{rt}] = 1 - [dx_{wd},\ dy_{wd}]/366$$

The orientation is normalized as illustrated bellow (from [0, 2π] to [-1, 1]):

<div align="center">
<img width="360px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/orientation_normalization.svg">
</div>

The velocity and the LiDAR is normalized by deviding their maxmum value respectively. We employ such normalized relative state representation for two reason. Fisrt, empirically, the input variable of magnitute between [-1,1] could accelerate the comvergence speed of neural networks. Second, as well as the most prominent reason, the relative coordinate frame could fundamentally improve the generalization ability of the RL model. That is, even we train the RL model in a fixed manner (start from the bottom left corner, end at the upper right corner), the trained model is capable of handling any start-end scenarios as long as their distance is within the maxmum planning distance **D** in the trainning phase, as showes bellow:

<div align="center">
<img width="360px" height="auto" src="https://github.com/XinJingHao/Sparrow-V0/blob/main/Imgs/state_eval.svg">
</div>
