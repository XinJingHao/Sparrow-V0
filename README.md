
<div align="center">
  <a ><img width="300px" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/LOGO%20sparrow.svg"></a>
</div>

## A Reinforcement Learning Friendly Simulator for Mobile Robot

<div align="center">
<img width="100%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/Show_SparrowV0.gif">
</div>

![Python](https://img.shields.io/badge/Python-blue)
![DRL](https://img.shields.io/badge/DRL-blueviolet)
![Mobile Robot](https://img.shields.io/badge/MobileRobot-ff69b4)

## Features

- **[Vectorizable](https://www.gymlibrary.dev/content/vectorising/)** (Fast data collection; Single environment is also supported)
- **[Domain Randomization](https://arxiv.org/pdf/1703.06907.pdf%60)** (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, sensor noise, magnitude of noise, maps)
- **Lightweight** (30kb, pure Python files. Only import, never worry about installation)
- **Accept image as maps** (Customize your own environments easily and rapidly)
- **Ubuntu/Windows** are both supported
- **Standard Gym API with both [Pytorch](https://pytorch.org/)/[Numpy](https://numpy.org/) data flow**
- **GPU/CPU are both acceptable** (If you use Pytorch to build your RL model, you can run you RL model and Sparrow both on GPU. Then you don't need to transfer the transitions from CPU to GPU anymore.)

## Installation

The dependencies for Sparrow are:

```bash
torch >= 1.13.1
gym >= 0.26.2
pygame >= 2.1.2
numpy >= 1.23.4
```
You can install **torch** by following the guidance from its [Official website](https://pytorch.org/get-started/locally/). We strongly suggest you install the **CUDA 11.7** version.

Then you can install **gym**, **pygame**, **numpy** via:

```bash
pip3 install gym==0.26.2 pygame==2.1.2 numpy==1.23.4
```

## Quick Start

After installation, you can play with Sparrow with your keyboard (up/down/left/right button) to test if you have installed it successfully:

```bash
python play_with_keyboard.py
```

## Train a DDQN model with Sparrow
The Sparrow is mobile robot simulator mainly designed for Deep Reinforcement Learning. In this section, we have prepared two python scripts to show you how to train a [DDQN](https://ojs.aaai.org/index.php/AAAI/article/download/10295/10154) model (Other Pytorch implementation of popular DRL algorithm is available at [here](https://github.com/XinJingHao/RL-Algorithms-by-Pytorch)) with **single** Sparrow and **vectorized** Sparrow. 

### Start training:
To train a DDQN model with **single** Sparrow, you can run:
```bash
python train_DDQN_single.py
```

In order to facilitate data collection and take full advantage of the parallel computing of GPU, you can use the vectorized version of Sparrow. To train a DDQN model with **vectorized** Sparrow, you can run:
```bash
python train_DDQN_vector.py
```

By default, the above two scripts will run on your GPU, if GPU is accessible, otherwise CPU. Specifically, in `train_DDQN_vector.py`, the Sparrow is vectorized by 10 copies. If this exceeds your maximum GPU memories, you can reduce the copies by setting fewer `actor_envs`:
```bash
python train_DDQN_vector.py --actor_envs 5
```

### Visualize the training curve:

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/ep_r.svg" align="right" width="35%"/>

For both scripts, we have incorporated **tensorboard** to visualize the training curve, as shown in the right. To enable it, you can just set the `write` to True, e.g.
```bash
python train_DDQN_single.py --write True
```
The training curve will be saved at the `runs` folder, for more details about how to install and use tensorboard, please click [here](https://pytorch.org/docs/stable/tensorboard.html). Also, you are free to use your own data visualization tools by modifying the scripts.

### Play with trained model:
During training, the DDQN model will be saved at the `model` folder every 10k steps (e.g. `model/10k.pth`). After training, you can play with it via:
```bash
python train_DDQN_single.py --render True --Loadmodel True --ModelIdex 10
```

## Dive into Sparrow

### Create your first env:

Since Sparrow has standard [Gym](https://www.gymlibrary.dev/) API, you can create the Sparrow environment via:

```python
import SparrowV0
import gym

env = gym.make('Sparrow-v0')
```

The above command will create a standard (single) Gym environment, and you can interact with it via:

```python
s, info = env.reset()
done = False
while not done:
    a = model(s) # your own RL model
    s_next, r, terminated, truncated, info = env.step(a)
    done = terminated + truncated
```

As we have mentioned, Sparrow is **vectorizable**, so you can create a vectorized environment via:

```python
import SparrowV0
import gym

if __name__ == '__main__':
    N = 4
    envs = gym.vector.AsyncVectorEnv([lambda: gym.make('Sparrow-v0', np_state = True) for _ in range(N)])
```

Here, `N`is the number of vectorized environments. In this context, the RL model should interact with the environment in a batched manner. And the dimension of **s, a, r, terminated, truncated** are **(N,32), (N,), (N,), (N,), (N,)** respectively. Note that **np_state=Ture** means the state will be returned in *numpy.narray*. More parameter setting will be introduced in the next section.

### Basic parameters:

There are 9 parameters could be configured when creating Sparrow:

```python
env = gym.make('Sparrow-v0',dvc, ld_num, np_state, colorful, state_noise, render_mode, render_speed, evaluator_mode, eval_map)
```

- **dvc (string; default `'cpu'`)**:
  
  - The device that runs the Sparrow
    
  - Should be one of `'cpu'`/`'cuda:0'`. We suggest using `'cuda:0'` (GPU) to accelerate simulation
    
- **ld_num (int; default `27`)**: 
  - number of LiDAR rays.
  
- **np_state (bool; default `False`)**:
  
  - `False`: return state in *torch.tensor* (will be on same device as dvc)
    
  - `True`: return state in *numpy.ndarray* (will be on CPU)
    
  - When using vectorized Env, np_state should be True, because gym.vector only support data in *numpy.ndarray*
    
- **colorful (bool; default `False`)**:
  - if `True`, the follow items will be randomized:
  
    - physical parameters (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, sensor noise, magnitude of noise)
    
    - initial position of the robot at the beginning of each episode
    
    - maps
    
- **state_noise (bool; default `False`)**:
  - if `True`: the state of the robot will contain uniformly distributed noise (it doesn't impact the accuracy of simulation)
  
- **render_mode (string; default `None`)**: 
  
  - `"human"`: render in a pygame window
    
  - `"rgb_array"`: *env.render()* will return the airscape of the robot in *numpy.ndarray*
    
  - `None`: not render anything
    
- **render_speed (string; default `'fast'`)**:
  -  control the rendering speed, should be one of `'fast'`/`'real'`/`'slow'`
  
- **evaluator_mode (bool; default `False`)**: 
  - if `True`, *env.reset()* will not swap maps and robot will always be initialized in bottom left corner.
  
- **eval_map (string; default `None`)**:
  -  if *evaluator_mode=True*, you need to designate the map on which you want to evaluate. And *eval_map* should be its absolute address, e.g. `os.getcwd()+'SparrowV0/envs/train_maps/map4.png'`
  

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/coordinate_frames.svg" align="right" width="25%"/>

### Coordinate Frames:

There are three coordinate frames in Sparrow as shows right. The ground truth position of the robot is calculated in **World Coordinate Frame**, which will be normalized and represented in **Relative Coordinate Frame** to comprise the RL state variable. The **Grid Coordinate Frame** comes from *pygame*, used to draw the robot, obstacles, target area, etc.


### Basic Robot Information:

The LiDAR perception range is 100cm×270°, with accuracy of 3 cm. The radius of the robot is 9 cm, and its collision threshold is 14 cm. 

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/basic_robot_info.svg" align="right" width="32%"/>

The maximum linear and angular velocity of the robot is 18 cm/s and 1 rad/s, respectively. The control frequency of the robot is 10Hz. And we use a simple but useful model to describe the kinematics of the robot

$$[V^{t+1}_{linear},\ V^{t+1}_{angular}] = K·[V^{t}_{linear},\ V^{t}_{angular}]+(1-K)·[V^{target}_{linear},\ V^{target}_{angular}]$$

Here, **K** is a hyperparameter between (0,1), describing the combined effect of inertia and friction, default: 0.6. The parameters mentioned in this section can be found in the *Robot initialization* and *Lidar initialization* part of `SparrowV0/envs/sparrow_v0.py` and customized according to your own scenario.



### RL representation:

The basic task in Sparrow is about driving the robot from the start point to the end point without colliding with the obstacles as fast as possible. To this end, in the following sub sections, we will define several basic components in Markov Decision Process.

#### State:

The state of the robot is a vector of length 32, containing **position** (*state[0:2] = [dx,dy]*), **orientation** (*state[2]=α*), **Velocity** (*state[3:5]=[v_linear, v_angular]*), **LiDAR** (*state[5:32] = scanning result*). The **position** and **orientation** are illustrated as bellow(left). These state variables will be normalized into the Relative Coordinate Frame before fed to the RL model. The velocity and the LiDAR are normalized by dividing their maximum value respectively. The position is normalized through $[dx_{rt},\ dy_{rt}] = 1 - [dx_{wd},\ dy_{wd}]/366$. And the orientation is normalized as illustrated bellow (right):

<div align="center">
<img width="40.5%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/state_train.svg">
<img width="38%" height="auto" src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/orientation_normalization.svg">
</div>

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/state_eval.svg" align="right" width="32%"/>

We employ such normalized relative state representation for two reason. First, empirically, the input variable of magnitude between [-1,1] could accelerate the convergence speed of neural networks. Second, as well as the most prominent reason, the relative coordinate frame could fundamentally improve the generalization ability of the RL model. That is, even we train the RL model in a fixed manner (start from the lower left corner, end at the upper right corner), the trained model is capable of handling any start-end scenarios as long as their distance is within **D** (the maximum planning distance in training phase), as illustrated right:


#### Action:
There are 6 discrete actions in Sparrow, controlling the target velocity of the robot:
- **Turn Left:** [ 0.36 cm/s, 1 rad/s ]
- **Turn Left + Move forward:** [ 18 cm/s, 1 rad/s ]
- **Move forward:** [ 18 cm/s, 0 rad/s ]
- **Turn Right + Move forward:** [ 18 cm/s, -1 rad/s ]
- **Turn Right:** [ 0.36 cm/s, -1 rad/s ]
- **Stop:** [ 0 cm/s, 0 rad/s ]

We strongly suggest not using the **Stop** action when training a RL model, because it may result the robot in standing still and generate low quality data. You might have also noted that when the robot is turn left or right, we also give it a small linear velocity. We do this to help the robot escape from the deadlock.

#### Reward:
If the robot collided with the obstacle, it would receive a negative reward of -10 and the episode would be terminated. If the robot arrived at the target area (the upper right corner of the map), it would receive a positive reward of 75 and the episode would be terminated as well. Otherwise, the robot would receive a reward according to:
$$reward = 0.3r_{d1} + 0.1r_{d2}+0.3r_v+0.3r_{\alpha}+0.1r_{d}$$
where $r_{d1}$ and $r_{d2}$ are negatively correlated to $d1$ and $d2$ (recall the illustration in the **State** subsection), with the maximum value of 1; $r_v$=1 if the linear velocity of the robot exceeds half of its maximum linear velocity, otherwise 0; $r_{\alpha}$ is negatively correlated to the absolute value of $\alpha$; $r_d$=-1 if the closest distance between robot and obstacle is smaller than 30 cm, otherwise 0.

#### Termination:
The episode would be terminated only when the robot collides with the obstacles or reaches the target area.

#### Truncation:
The episode would be truncated if one of these situations happens:
- the episode steps exceed 2000
- the robot rotates more than 1.5 times in place (to prevent the robot from generating low quality data)

### Improving the generalization ability:
In this section, we will introduce four tips incorporated in Sparrow that could improve the generalization ability of the trained model.

#### Random maps:
To prevent the robot from overfitting in one specific map, we have prepared 16 different training maps in `SparrowV0/envs/train_maps`. While training, these maps will swap automatically according to the `self.swap_ferq` in `sparrow_v0.py`. This mechanism can be disabled by setting `evaluator_mode=True`. In this case, you have to designate the map you would like to use by passing the absolute address of the map to `eval_map`.

#### Random obstacles:
You might have noticed that the `SparrowV0/envs/train_maps/map0.png` contains no obstacle, so we will generate some random obstacles when using this map. For more details, please check the `self._generate_obstacle()`  in `sparrow_v0.py`. These random obstacles could constitute a more diverse state space so as to boost the generalization ability.

#### Random initialization:
The initial place of the robot could be randomized so that more state space could be explored. To this end, the robot will be randomly initialized according to the map related files in `SparrowV0/envs/train_maps_startpoints`. We have already prepared them for you! But you can also customize your own random initialization files using `SparrowV0/envs/generate_startpoints.py`. 
Steps:
- open `generate_startpoints.py` and set the map you are going to working on at `main(map_name='map1')`
- run `generate_startpoints.py` and use the left mouse button to designate random initialization points
- press the `Esc` button to save these points, which would be saved in `SparrowV0/envs/train_maps_startpoints` with the same name as the map in `.npy` format.

#### Domain randomization:
[Domain randomization](https://arxiv.org/pdf/1703.06907.pdf%60) has been proven to be an effective method for generalizing model trained in simulation to the real world, and has been elegantly incorporated in Sparrow, taking full advantage of its vectorizable feature. You can enable Domain randomization by creating vectorized Sparrow and set `colorful` and `state_noise` to be True, and the simulation parameters (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, sensor noise, magnitude of noise, maps) would be randomly generated in each stream of the vectorized Sparrow environment.

### Simulation Speed:
If `render_mode=None` or `render_mode="rgb_array"`, Sparrow would run at its maximum simulation speed (depending on the hardware). However, if `render_mode="human"`, there would be three options regarding the simulation speed:
- `render_speed == 'fast'`: render the Sparrow in a pygame window with maximum FPS
- `render_speed == 'slow'`: render the Sparrow in a pygame window with 5 FPS. Might be useful when debugging.
- `render_speed == 'real'`: render the Sparrow in a pygame window with **1/ctrl_interval** FPS, in accordance with the real world speed.

### Customize your own maps:
Sparrow takes `.png` images as its maps, e.g. the `map0.png`~`map15.png` in `SparrowV0/envs/train_maps/`. Therefore, you can draw your own maps with any image process software easily and conveniently, as long as it satisfies the following requirements:
- saved in `.png` format
- resolution (namely the map size) equals to 366×366
- obstacles are in black (0,0,0) and free space is in white (255,255,255)
- adding a fence to surround the map so that the robot cannot run out of the map

**Important:** please do not delete or modify the `SparrowV0/envs/train_maps/map0.png`

### AutoReset:
The environment copies inside a vectorized environment may be done (terminated or truncated) in different timesteps. Consequently, it is inefficient or even improper to call the *env.reset()* function to reset all copies whenever one copy is done, necessitating the design of **AutoReset** mechanism. Since the Sparrow inherits the `gym.vector`, it also inherits its AutoReset mechanism. That is, whenever the *env.step()* leads to termination or truncation, it will reset the current episode immediately and output the reset state (rather than the next state), as illustrated bellow:

<img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V0/AutoReset.svg" align="center" width="100%"/>

## Citing the Project

To cite this repository in publications:

```bibtex
@misc{Sparrow_v0,
  author = {Jinghao Xin},
  title = {Sparrow: A Reinforcement Learning Friendly Simulator for Mobile Robot},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XinJingHao/Sparrow-V0}},
}
```
## Writing in the end
The name "Sparrow" actually comes from an old saying *“The sparrow may be small but it has all the vital organs.”* Hope you enjoy using Sparrow! 

Additionally, we have made detailed comments on the source code (`SparrowV0/envs/sparrow_v0.py`) so that you can modify Sparrow to fit your own problem. But only for non-commercial purposes, and all rights are reserved by [Jinghao Xin](https://github.com/XinJingHao).
