import gym
import torch
import pygame
import numpy as np
from collections import deque
import copy
import os


'''
Sparrow: A Reinforcement Learning Friendly Simulator for Mobile Robot

Several good features:
· Vectorizable (Fast data collection; Single environment is also supported)
· Domain Randomization (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, 
  sensor noise, magnitude of noise and maps can be randomized while training)
· Lightweight (Consume only 150~200 mb RAM or GPU memories per environment)
· Standard Gym API with both Pytorch/Numpy data flow
· GPU/CPU are both acceptable (If you use Pytorch to build your RL model, you can run your RL model and Sparrow both on 
  GPU. Then you don't need to transfer the transitions from CPU to GPU anymore.)
· Easy to use (30kb pure Python files. Just import, never worry about installation)
· Ubuntu/Windows are both supported
· Accept image as map (Customize your own environments easily and rapidly)
· Detailed comments on source code

Only for non-commercial purposes.
All rights reserved. 

“The sparrow may be small but it has all the vital organs.”
Developed by Jinghao Xin. Github：https://github.com/XinJingHao
2023/2/21

Current version: 2023/6/20
'''

# Color of obstacles when render
_OBS = (64, 64, 64)

class SparrowV0Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, dvc='cpu', ld_num=27, np_state = False, colorful = False, state_noise=False, render_mode=None, render_speed='fast', evaluator_mode = False, eval_map=None):
        self.dvc_name = dvc # 'cpu' or 'cuda:0'. Better use 'cuda:0' to accelerate simulation.
        self.dvc = torch.device(dvc)# running device of Sparrow, better use GPU to accelerate simulation

        self.np_state = np_state
        # False: return state in torch.tensor(will be in same device as self.dvc); True: return state in numpy.ndarray
        # When using vectorized Env, it should be True, because gym.vector only support data in numpy.ndarray.

        self.state_noise = state_noise
        # if True, the state will contain uniformly distributed noise (it doesn't impact the accuracy of simulation)

        self.colorful = colorful
        # if True, the follow items will be randomized:
        # maps (will be swaped according to self.swap_ferq. Note that the number and type of obstacles in map0 will also be randomized)
        # initial position of the robot at the beginning of each episode
        # physical parameters (control interval, control delay, max linear velocity, max angular velocity, inertia, friction, magnitude of noise)

        self.evaluator_mode = evaluator_mode
        # if True, reset() will not swap maps and robot always inits in bottom left corner.

        ''' Map initialization: map_idx // map // random_start/start_points '''
        if self.evaluator_mode:  # used to evaluate a single map
            self.random_start = False # robot always inits in bottom left corner.
            self.map = pygame.image.load(eval_map) # load map into pygame, eval_map should be the address of one map
            self.map_idx = 0 if eval_map[-8:] == 'map0.png' else 1 # index of the eval map. We only distinguish 0 and 1 in evaluator_mode.
        else:
            # we only init self.maps here. map_idx // map // random_start/start_points will be generated at self.reset()
            # ['map0.png' 'map1.png' 'map10.png' 'map11.png' ... 'map14.png' 'map15.png' 'map2.png' 'map3.png' ... 'map9.png']
            self.maps = np.sort(os.listdir(os.getcwd() + '/SparrowV0/envs/train_maps'))
        self.random_start_rate = 1.0

        ''' Pygame initialization '''
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode # should be one of "human", "rgb_array", None
        # "human": will render in a pygame window
        # "rgb_array" : calling env.render() will return the airscape of the robot in numpy.ndarray
        # None: not render anything

        self.window_size = 366 # size of the map
        self.window = None
        self.clock = None
        self.canvas = None
        self.render_speed = render_speed # should be one of 'real', 'fast', 'slow'

        self.target_area = 100  # enter the target_area×target_area space in the upper right of the map will be considered win.
        self.sqrt2 = np.sqrt(2)
        self.total = self.window_size * self.sqrt2
        self.AWARD = 75 # reward of enter the target_area
        self.PUNISH = -10 # negtive reward when collision happens
        self.min_ep_steps = 150 # min episode steps, for autonomous episode truncation
        self.ep_step = 0 # record the current steps of each episode
        self.ep_counter = 0 # record how many episodes have last after last map swapping; used in conjunction with swap_ferq
        self.swap_ferq = 100  # map swap frequency of each env, in episode

        ''' Robot initialization '''
        self.car_radius = 9  # cm
        self.collision_trsd = self.car_radius+5 # collision threshould, in cm
        self.ctrl_interval = 0.1 + 0.02*self.np_random.random()*self.colorful  # control interval, in second
        self.ctrl_delay = int(0 + self.np_random.integers(0,2)*self.colorful) # control delay, in ctrl_interval
        self.ctrl_pipe = deque(maxlen=self.ctrl_delay+1) # holding the delayed action
        self.v_linear_max = 18 + self._random_noise(2)*self.colorful # max linear velocity, in cm/s
        self.v_angular_max = 1 + self._random_noise(0.1)*self.colorful  # max angular velocity, in rad/s
        self.a_space = torch.tensor([[0.2*self.v_linear_max , self.v_angular_max],[self.v_linear_max , self.v_angular_max],
                                     [self.v_linear_max, 0], # v_linear, v_angular
                                     [self.v_linear_max, -self.v_angular_max],[0.2*self.v_linear_max, -self.v_angular_max],
                                     [0,0]], device=self.dvc) # a_space[-1] is no_op，agent is not allowed to use.
        self.car_init_min = 40 # robot initial area when random_start is False. x_wd/ y_wd ∈ [car_init_min,car_init_max]
        self.car_init_max = 60 # robot initial area when random_start is False. x_wd/ y_wd ∈ [car_init_min,car_init_max]
        self.car_init_or = 2*torch.pi  # robot orientation initial interval when random_start is False. [0, car_init_or]
        self.K = 0.6 + self._random_noise(0.05)*self.colorful # Kinematic parameter of the robot. Representing inertia and friction
        self.car_state_upperbound = torch.tensor([self.window_size, self.window_size, 1 ,
                                                  self.v_linear_max, self.v_angular_max], device=self.dvc)
        # car_state_upperbound is used to normalize the robot's state.
        # Note that orientation need normalization separately by _angle_abs2relative(), so we give it '1' here, see self._get_obs()

        ''' Auto truncation initialization '''
        # used to prevent robot from circling around and thus generating low quality data
        self.trunc_len = int(1.5 * 2*torch.pi/self.v_angular_max/self.ctrl_interval) # steps needed for turn 1.5 circles
        self.v_linear_buffer = deque(maxlen=self.trunc_len)
        self.auto_coef = 0.92 # for oldversion, auto_coef=1.0; one can get a more frequent truncation mechanism by using smaller auto_coef


        ''' Lidar initialization '''
        self.ld_acc = 3 # lidar scan accuracy (cm). Reducing accuracy can accelerate simulation;
        self.ld_num = ld_num # number of lidar streams
        self.ld_range = 100 # max scanning distance of lidar (cm). Reducing ld_range can accelerate simulation;
        self.ld_scan_result = torch.zeros(self.ld_num, device=self.dvc) # used to hold lidar scan result
        self.ld_angle_interval = torch.arange(self.ld_num, device=self.dvc) * 1.5 * torch.pi / (self.ld_num-1)  - 0.75*torch.pi

        ''' Robot/Lidar perception noise(uniform) initialization '''
        if self.state_noise:
            self.car_sate_nis_mag = torch.tensor([1,1,torch.pi/50,0.2,torch.pi/100],device=self.dvc) # magnitude of robot pose noise (before normalization)
            self.ld_nis_mag = 1 # lidar perception noise(before normalization). Due to the repercussion of ld_acc, there is no need to set ld_nis_mag to large.
            if self.colorful: # diverse the magnitude of noise.
                self.car_sate_nis_mag *= self.np_random.random()
                self.ld_nis_mag *= self.np_random.random()

        ''' Obstacle initialization '''
        self.obs_canvas = pygame.Surface((self.window_size, self.window_size))
        self.bold = 2*self.ld_acc # line width of the bounding box of obstacles。越粗，雷达越不容易射穿。 在cpu上运行时，增大bold掉帧严重；gpu上增大bold帧率变化不大。

        ''' State/Action space initialization '''
        self.observation_num = 5+self.ld_num
        self.action_num = len(self.a_space)-1 # a_space[-1] is no_op, agent is not allowed to use.
        self.observation_space = gym.spaces.Box(-np.ones(self.observation_num,dtype=np.float32),
                                                np.ones(self.observation_num,dtype=np.float32))
        self.action_space = gym.spaces.Discrete(self.action_num)
        gym.logger.set_level(40)


        assert self.v_linear_max*self.ctrl_interval < self.collision_trsd-self.car_radius
        assert self.ld_acc < self.collision_trsd
        assert self.min_ep_steps > self.trunc_len

    def _random_noise(self,magnitude):
        '''Generate random noise in magnitude*[-1,1) with self.np_random'''
        return (self.np_random.random()-0.5)*2 * magnitude

    def _world_2_grid(self, location_wd):
        ''' Convert world coordinates (denoted by _wd, continuous, unit: cm) to grid coordinates (denoted by _gd, discrete, 1 grid = 1 cm)
            Input: torch.tensor; Output: torch.tensor; Shape: [[x0,y0],[x1,y1]...[xn,yn]] or [x0,y0] '''
        if len(location_wd.shape)<2:
            p = location_wd.unsqueeze(0).clone() # for [x0,y0]
        else:
            p = location_wd.clone()  # for [[x0,y0],[x1,y1]...[xn,yn]]
        p[:, 1] = self.window_size - p[:, 1]
        return p.round().int()

    def _angle_abs2relative(self, a_theta):
        ''' Convert absolute orientation([0,2pi], in _wd frame) to relative orientation([-1,1], in _rt frame)
            parallelly oriented to the target: r_theta = 0  
            parallelly backward to the target: r_theta = 1(right side); r_theta = -1(right side)'''
        r_theta = a_theta / np.pi
        if r_theta > 1.25:
            r_theta = 2.25 - r_theta
        elif r_theta > 0.25:
            r_theta = 0.25 - r_theta
        else:
            r_theta = 0.25 - r_theta
        return r_theta

    def _ld_scan(self):
        '''Get the scan result of lidar. How to accelerate the scan process?'''
        # 扫描前首先同步雷达与小车位置:
        self.ld_angle = self.ld_angle_interval + self.car_state[2]# 雷达-小车方向同步
        self.ld_vectors_wd = torch.stack((torch.cos(self.ld_angle), torch.sin(self.ld_angle)), dim=1)  # 雷达射线方向
        self.ld_end_wd = self.car_state[0:2] + self.car_radius * self.ld_vectors_wd  # 扫描过程中，雷达射线末端世界坐标(初始化于小车轮廓)
        self.ld_end_gd = self._world_2_grid(self.ld_end_wd)  # 扫描过程中，雷达射线末端栅格坐标

        # 扫描初始化
        self.ld_scan_result *= 0  # 结果归零
        increment = self.ld_vectors_wd * self.ld_acc  # 每次射出的增量
        if self.dvc_name == 'cpu':
            _in = ((self._world_2_grid(self.car_state[0:2]) - self.bound_gd) ** 2).sum(dim=-1) < (self.ld_range + self.bold + 1)**2 # 查看哪些bound点在感知范围内
            bound_gd_in, bound_num_in = self.bound_gd[_in], _in.sum() # 可能会扫描到的bound点，以及他们的数量
        else: #在cpu上运算时，做排除法后再扫描更快，而GPU直接扫描更快
            bound_gd_in, bound_num_in = self.bound_gd, self.bound_num # 可能会扫描到的bound点，以及他们的数量

        # 烟花式扫描
        for i in range( int((self.ld_range-self.car_radius)/self.ld_acc) + 2 ): # 多扫2次，让最大值超过self.ld_range，便于clamp
            # 更新雷达末端位置
            
            # 判断ld_end_gd的坐标是否在bound_gd中, eg: ld_end_gd=[[1,2], [3,4], [5,6]];bound_gd=[[0,0],[5,6],[1,1],[2,2],[3,3],[1,2],[4,4]];goon=[True, False, True]
            pre_goon = (self.ld_end_gd.unsqueeze(1).repeat(1, bound_num_in, 1) - bound_gd_in).bool()  # 将雷达端点坐标扩充为物体边界栅格总数，相减，等于0处表示ld坐标在bound坐标里
            # pre_goon = (self.ld_end_gd[:,None,:] - self.bound_gd).bool() # 可以替换上一行，GPU上会快一些，CPU上会慢一些
            goon = torch.all(torch.any(pre_goon, dim=-1), dim=-1) # 通过一系列与、或操作，获得最终结果
            self.ld_end_wd += (goon.unsqueeze(-1) * increment)  # 更新雷达末端世界坐标,每次射 ld_acc cm
            self.ld_end_gd = self._world_2_grid(self.ld_end_wd)# 更新雷达末端栅格坐标（必须更新，下一轮会调用）
            self.ld_scan_result += (goon * self.ld_acc)# 累计扫描距离

            if (~goon).all(): break # 如果所有ld射线都被挡，则扫描结束

        # 扫描的时候从小车轮廓开始扫的，最后要补偿小车半径的距离; (ld_num, ); torch.tensor
        return (self.ld_scan_result + self.car_radius).clamp(0,self.ld_range)


    def _get_obs(self):
        '''Get the observation of the robot.
           Return: normalized [x, y, theta, v_linear, v_angular, lidar_results(0), ..., lidar_results(26)] '''
        self.ld_result = self._ld_scan() # Get the scan result of lidar
        if self.render_mode is not None:
            self.ld_result_cpu = self.ld_result.cpu() # used to draw the lidar ray

        if self.state_noise:
            # add noise to state, and normalization (orientation will be separately normalized by _angle_abs2relative)
            car_state = (self.car_state + self.car_sate_nis_mag * (torch.rand(5,device=self.dvc)-0.5) * 2) / self.car_state_upperbound
            ld_result = (self.ld_result + self.ld_nis_mag * (torch.rand(self.ld_num,device=self.dvc)-0.5) * 2) / self.ld_range
        else:
            # normalization (orientation will be separately normalized by _angle_abs2relative)
            car_state = self.car_state / self.car_state_upperbound
            ld_result = self.ld_result / self.ld_range
        car_state[0:2] = 1 - car_state[0:2]  # convert normalized [x_wd,y_wd] to [x_rl,y_rl]

        car_state[2] = self._angle_abs2relative(car_state[2])  # normalize the orientation from [0,2pi] to [-1,1]

        if self.np_state:
            return torch.concat((car_state, ld_result)).cpu().numpy() # return in numpy.narray
        else:
            return torch.concat((car_state, ld_result)) # return in torch.tensor on self.dvc


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # seed self.np_random
        self.ep_step = 0

        '''Step1: initialize maps --> self.map_idx // self.map // self.random_start/self.start_points'''
        # swap the training map according to swap_ferq, ensuring 'colorful' covers all the maps.
        if (not self.evaluator_mode) and (self.ep_counter % self.swap_ferq == 0):
            if self.colorful:
                # at least 10% maps are map0.png(with random obstacles), more robust for trainning
                if self.np_random.random() < 0.1: self.map_idx = 0
                else: self.map_idx = self.np_random.integers(0, len(self.maps))

            else:
                self.map_idx = 0

            # load map_.png into pygame according to the index of map (map_idx)
            self.map_name = self.maps[self.map_idx]
            self.map = pygame.image.load(os.getcwd() + '/SparrowV0/envs/train_maps/' + self.map_name)

            # inquire whether the current map support random_start,
            # that is whether the current map has its corresponding train_maps_startpoints file
            try:
                self.start_points = np.load(os.getcwd() + '/SparrowV0/envs/train_maps_startpoints/' + self.map_name[0:-4]+'.npy')
                self.random_start = True # The corresponding file is found and can be initialized randomly

                # flip the map with probability of 0.5 to enhance generalization ability
                if self.np_random.random() < 0.5:
                    # operate the map
                    self.map = pygame.transform.flip(self.map, False, True)
                    self.map = pygame.transform.rotate(self.map, 90)
                    # operate the corresponding random start points to match the flipped map
                    self.start_points[:, 0], self.start_points[:, 1] = self.start_points[:, 1].copy(), self.start_points[:, 0].copy()
                    self.start_points[:, 2] = (2.5 * np.pi - self.start_points[:, 2]) % (2 * np.pi)
            except:
                self.random_start = False

        '''Step2: initialize the car_state of robot'''
        # if random_start is supported, randomly init the pose of the robot with a linear decreasing probability
        if self.random_start and self.np_random.random() < self.random_start_rate:
            start_idx = self.np_random.integers(0, len(self.start_points))
            self.car_state = torch.tensor([self.start_points[start_idx,0], # x_wd
                                           self.start_points[start_idx,1], # y_wd
                                           self.start_points[start_idx,2], # orientation_wd
                                           0., # v_linear
                                           0.],# v_angular
                                           device=self.dvc, dtype=torch.float32)
        else:# init the pose of the robot in the bottom left corner
            self.car_state = torch.tensor([self.np_random.integers(self.car_init_min,self.car_init_max), # x_wd
                                           self.np_random.integers(self.car_init_min,self.car_init_max), # y_wd
                                           self.np_random.random()*self.car_init_or, # orientation
                                           0., # v_linear
                                           0.],# v_angular
                                           device=self.dvc)

        '''Step3: clear ctrl_pipe with [0,0]'''
        for i in range(self.ctrl_delay+1):
            self.ctrl_pipe.append(len(self.a_space)-1)

        '''Step4: obstacle reset(based on self.map)'''
        self._generate_obstacle()

        '''Step5: generate initial observation'''
        observation, info = self._get_obs(), {"info":None}

        # render
        if self.render_mode == "human":
            self._render_frame()

        self.ep_counter += 1
        return observation, info

    def _generate_obstacle(self):
        '''load map, draw obstacles, and save its grid coordinates into bound_gd.
            the bound_gd will be used when lidar scanning'''

        # load map into obstacle canvas
        self.obs_canvas.blit(self.map, self.obs_canvas.get_rect())

        # draw target area (only when rendering)
        if self.render_mode == "human":
            pygame.draw.rect(self.obs_canvas,(200, 255, 200),
                             pygame.Rect((self.window_size-self.target_area - self.bold+2, self.bold+1),(self.target_area-3, self.target_area-3)))

        # map0.png contains no obstacle itself, so we need to add random obstacles for it
        if self.map_idx == 0:
            obs_num = int(3 + self.np_random.integers(0, 4) * self.colorful)  # number of obstacles
            obs_min = 25
            for _ in range(obs_num):
                theta = self.np_random.random() * 2 * np.pi # orientation
                w, h = self.np_random.integers(obs_min, 2 * obs_min), self.np_random.integers(obs_min, 2 * obs_min) # size

                # area for placing the obstacles
                x = self.np_random.integers(100 + obs_min, self.window_size - obs_min)
                y = self.np_random.integers(obs_min, self.window_size - self.target_area - obs_min)
                if _ % 2 == 0: x, y = y, x

                type = 0 + self.colorful * self.np_random.integers(0, 4) # type of obstacles（0=□；1=X；2=L；3=H;4=U）
                apex = self._generate_obs_apex(x, y, w, h, theta, type) # apexes of obstacles

                # draw the random obstacles
                for i in range(len(apex)):
                    pygame.draw.line(self.obs_canvas, (0, 0, 0), apex[i, 0], apex[i, 1], width=self.bold)

        # convert canvas to numpy.narray
        obstacle_canvas = pygame.surfarray.pixels3d(self.obs_canvas)[:,:,0]

        # calculate and save the grid coordinates of the obstacles
        x_, y_ = np.where(obstacle_canvas == 0)
        self.bound_gd = torch.tensor(np.stack((x_,y_), axis=1), device=self.dvc)
        self.bound_num = len(self.bound_gd)

    def _generate_obs_apex(self,x0,y0,w,h,theta,type=0):
        '''
        Input: 
            coordinate ([x0,y0] in _wd frame) of the centre point of a obstacle, 
            width(w in cm), height(h in cm), orientation(theta in _wd frame)
            type （0=□；1=X；2=L；3=H;4=U）
        Output:
            line apexes of the obstacle (in _gd frame)
        '''
        theta_br, theta_tr, theta_tl = theta, theta + np.arctan(w/h), theta + np.pi/2
        diagonal = np.sqrt(w**2 + h**2)
        # calculate the apexes in _wd frame, and transfer them to _gd frame by .round().astype(int)
        bl = np.array([x0,self.window_size-y0]).round().astype(int)  #[x,y]
        br = np.array([x0+h*np.cos(theta_br),self.window_size-(y0+h*np.sin(theta_br))]).round().astype(int)
        tr = np.array([x0+diagonal*np.cos(theta_tr),self.window_size-(y0+diagonal*np.sin(theta_tr))]).round().astype(int)
        tl = np.array([x0+w*np.cos(theta_tl),self.window_size-(y0+w*np.sin(theta_tl))]).round().astype(int)

        if type == 0:
            apexes = np.array([[bl, br], [br, tr], [tr, tl], [tl, bl]])
        elif type == 1:
            apexes = np.array([[bl, tr], [tl, br]])
        elif type == 2:
            apexes = np.array([[bl, (br+bl)/2], [tl, bl]])
        elif type == 3:
            apexes = np.array([[bl, br],[tr, tl], [(bl+br)/2,(tr+tl)/2]])
        elif type == 4:
            apexes = np.array([[bl, br], [tl, bl], [tl, tr]])

        return apexes - (tr - bl) / 2 # shift the coordinates in centre manner


    def _Kinematic_model(self, a):
        ''' V_now = K*V_previous + (1-K)*V_target '''
        self.car_state[3:5] = self.K * self.car_state[3:5] + (1-self.K)*self.a_space[a]
        return torch.tensor([self.car_state[3], self.car_state[3], self.car_state[4]], device=self.dvc)


    def _reward_function(self):
        min_ld_result = self.ld_result.min().item()  # minimum distance to the obstacle
        car_state = self.car_state.cpu().numpy()  # x,y,theta,v_linear,v_angular ; unnormalized; in _wd frame

        if min_ld_result <= self.collision_trsd:
            # collide with the obstacle
            if self.render_mode == "human": print('Dead')
            return self.PUNISH, True

        elif (car_state[0:2] > self.window_size - self.target_area).all():
            # arrive at the target area
            if self.render_mode == "human": print('Win')
            return self.AWARD, True
        else:
            interal = (car_state[0] - car_state[1]) / self.sqrt2
            d1 = np.abs(interal)
            d2 = self.total - self.sqrt2 * car_state[0] + interal

            r_d1 = 100 ** (-d1 / (self.window_size/1.414)) # The closer to the diagonal, the greater the r_d1 (exponential); r_d1 in (0,1]
            r_d2 = self.target_area / np.clip(d2, self.target_area, self.total)  # The closer to the target area, the greater the r_d2; r_d2≤1

            r_v = car_state[3] > (0.5 * self.v_linear_max) # r_v = 0 or 1

            r_ag = 1 - np.abs(self._angle_abs2relative(car_state[2])) # r_ag∈[0,1]; orient the target,r_ag=1; back to the target,r_ag=0
            r_ag -= 0.2  # [0,1] to [-0.2,0.8], punish when back to the target
            if r_ag < 0: r_ag *= 80  # [-0.2,0.8] to [-16,0.8], increase the punishment, prevent circling around

            # punish when too close to the obstacle
            if min_ld_result < self.collision_trsd + 30:
                r_close = -(self.collision_trsd + 30 - min_ld_result) / 30  # [-1,0]
            else:
                r_close = 0

            reward = 0.1 * r_d2 + 0.3 * r_d1 + 0.3 * r_v + 0.3 * r_ag + 0.1 * r_close

            return reward, False

    def step(self,current_a):
        self.ep_step += 1

        # linear decreasing the random_start_rate of the robot
        if self.random_start_rate>0.4: self.random_start_rate -= 0.000015
        else: pass

        # control delay mechanism
        a = self.ctrl_pipe.popleft() # a is the delayed action
        self.ctrl_pipe.append(current_a) # current_a is the action mapped by the current state

        # calculate the current velocity based on the delayed action and the Kinematic_model
        velocity = self._Kinematic_model(a)

        # calculate the pose of the robot after executing the delayed action
        movement = velocity * torch.tensor([torch.cos(self.car_state[2]), torch.sin(self.car_state[2]), 1], device=self.dvc)
        self.car_state[0:3] += movement * self.ctrl_interval

        # keep the orientation between [0,2pi]
        if self.car_state[2] < 0: self.car_state[2] += 2 * torch.pi
        elif self.car_state[2] > 2 * torch.pi: self.car_state[2] -= 2 * torch.pi

        # auto truncation
        if not self.evaluator_mode:
            # put linear velocity into deque buffer
            self.v_linear_buffer.append(self.car_state[3].item())
            # if circling around steps exceeds trunc_len*auto_coef, truncate the current episode ## loose constraint, more truncation
            if (self.ep_step > self.min_ep_steps) \
                    and ((np.array(self.v_linear_buffer) < 0.5 * self.v_linear_max).sum() > self.auto_coef * self.trunc_len):
                truncated = True
            else:
                truncated = False
        else: truncated = False

        observation, info,  = self._get_obs(), {"info":None}
        reward, terminated = self._reward_function()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size , self.window_size ))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # init canvas
        if self.canvas is None :
            self.canvas = pygame.Surface((self.window_size , self.window_size ))

        # draw obstacles on canvas
        self.canvas.blit(self.obs_canvas, self.obs_canvas.get_rect())

        # draw lidar rays on canvas
        ld_result = self.ld_result_cpu.clone() # (ld_num, ), on cpu
        ld_real_sta_gd = self._world_2_grid(self.car_state[0:2]).cpu().numpy()
        ld_real_end_gd = self._world_2_grid(self.car_state[0:2].cpu() + ld_result.unsqueeze(-1) * self.ld_vectors_wd.cpu()).numpy()
        for i in range(self.ld_num):
            e = 255*ld_result[i]/self.ld_range
            pygame.draw.line(self.canvas, (255-e, 0, e), ld_real_sta_gd[0], ld_real_end_gd[i], width=2)

        # draw collision threshold on canvas
        pygame.draw.circle(
            self.canvas,
            _OBS,
            self._world_2_grid(self.car_state[0:2]).cpu().numpy()[0],
            self.collision_trsd,
        )

        # draw robot on canvas
        pygame.draw.circle(
            self.canvas,
            (200, 128, 250),
            self._world_2_grid(self.car_state[0:2]).cpu().numpy()[0],
            self.car_radius,
        )
        # draw robot orientation on canvas
        head = self.car_state[0:2].cpu() + self.car_radius * torch.tensor([torch.cos(self.car_state[2]), torch.sin(self.car_state[2])])
        pygame.draw.line(
            self.canvas,
            (0, 255, 255),
            self._world_2_grid(self.car_state[0:2]).cpu().numpy()[0],
            self._world_2_grid(head).numpy()[0],
            width=2
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            if self.render_speed == 'real':
                self.clock.tick(int(1 / self.ctrl_interval))
            elif self.render_speed == 'fast':
                self.clock.tick(0)
            elif self.render_speed == 'slow':
                self.clock.tick(5)
            else:
                print('Wrong Render Speed, only "real"; "fast"; "slow" is acceptable.')

        else: #rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

