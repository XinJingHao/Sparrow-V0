import SparrowV0
import gym
import copy
import torch
import torch.nn as nn
import numpy as np
import os, shutil
import argparse
from datetime import datetime
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def str2bool(v):
	'''transfer str to bool for argparse
	You can just ignore this funciton. It's not related to the RL and Sparrow.'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--write', type=str2bool, default=False, help='Whether use SummaryWriter to record the training curve')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Whether load pretrained model')
parser.add_argument('--ModelIdex', type=int, default=10, help='which model(e.g. 10k.pth) to load')

parser.add_argument('--Max_train_steps', type=int, default=2e5, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e4, help='Model saving interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='steps for random policy to explore')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.998, help='decay rate of explore noise')
opt = parser.parse_args()
opt.state_dim = 5+27 # [dx,dy,orientation,v_linear,v_angular] + [lidar result]
opt.action_dim = 5

if opt.write: from torch.utils.tensorboard import SummaryWriter

def main(opt):
	# init DDQN model
	if not os.path.exists('model'): os.mkdir('model')
	model = DDQN_Agent(opt)
	if opt.Loadmodel: model.load(opt.ModelIdex)

	if opt.render: # play with a specific map
		eval_map = os.getcwd() + '/SparrowV0/envs/train_maps/map0.png'
		eval_env = gym.make('Sparrow-v0', dvc=device, render_mode='human', evaluator_mode=True, eval_map=eval_map)
		while True:
			score = evaluate_policy(eval_env, model, 1)
			print('Score:', score)

	else: # trian with Sparrow
		if opt.write:
			# use SummaryWriter to record the training curve
			timenow = str(datetime.now())[0:-10]
			timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
			writepath = 'runs/Sparrow_v0' + timenow
			if os.path.exists(writepath): shutil.rmtree(writepath)
			writer = SummaryWriter(log_dir=writepath)

		# build environment and experience replay buffer
		env = gym.make('Sparrow-v0', dvc=device) # return data in torch.tensor on 'device'
		buffer = ReplayBuffer(opt.state_dim, max_size=int(min(opt.Max_train_steps, 1e6)))

		total_steps = 0
		while total_steps < opt.Max_train_steps:
			s, _ = env.reset()
			done, ep_r = False, 0
			while not done:
				if buffer.size < opt.random_steps:
					a = env.action_space.sample()
				else:
					a = model.select_action(s, deterministic=False)
				s_next, r, dw, tr, info = env.step(a) #dw(terminated): die or win; tr: truncated
				done = dw + tr
				buffer.add(s, a, r, s_next, dw) # the trainsition is already on 'device'!
				s = s_next
				ep_r += r
				total_steps += 1

				if total_steps >= opt.random_steps and total_steps % 50 == 0:
					for _ in range(50): model.train(buffer) # train 50 times every 50 steps
					if model.exp_noise > 0.01: model.exp_noise *= opt.noise_decay # e-greedy noise decay

				'''save model'''
				if (total_steps) % opt.save_interval == 0:
					model.save(int(total_steps/1e3))

			if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
			print('Sparrow-v0:', 'steps: {}k'.format(round(total_steps / 1000,2)), 'score:', ep_r)


class ReplayBuffer():
	'''Experience replay buffer'''
	def __init__(self, state_dim, max_size=int(1e6)):
		self.device = device
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		# With the aid of Sparrow's 'cuda:0'(torch) data flow, we are able to run RL model and ReplayBuffer directly on GPU.
		# Never transfer the data from CPU to GPU or from numpy to pytorch anymore !!!
		self.state = torch.zeros((max_size, state_dim),device=self.device)
		self.action = torch.zeros((max_size, 1),device=self.device,dtype=torch.int64)
		self.reward = torch.zeros((max_size, 1),device=self.device)
		self.next_state = torch.zeros((max_size, state_dim),device=self.device)
		self.dw = torch.zeros((max_size, 1),device=self.device,dtype=torch.bool)

	def add(self, state, action, reward, next_state, dw):
		# transitions are already on the same device as ReplayBuffer
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(low=0, high=self.size, size=(batch_size,))
		return self.state[ind], self.action[ind], self.reward[ind], self.next_state[ind], self.dw[ind]

class Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Q_Net, self).__init__()
		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, int(net_width/2))
		self.l3 = nn.Linear(int(net_width/2), action_dim)

	def forward(self, state):
		q = torch.relu(self.l1(state))
		q = torch.relu(self.l2(q))
		q = self.l3(q)
		return q

class DDQN_Agent(object):
	def __init__(self,opt,):
		self.q_net = Q_Net(opt.state_dim, opt.action_dim, opt.net_width).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.gamma = opt.gamma
		self.tau = 0.005
		self.batch_size = opt.batch_size
		self.exp_noise = opt.exp_noise
		self.action_dim = opt.action_dim

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = state.unsqueeze(0) # state from Sparrow are in 'cuda:0' already!
			if deterministic:
				a = self.q_net(state).argmax().item()
			else:
				if np.random.rand() < self.exp_noise:
					a = np.random.randint(0,self.action_dim)
				else:
					a = self.q_net(state).argmax().item()
		return a

	def train(self,replay_buffer):
		s, a, r, s_next, dw = replay_buffer.sample(self.batch_size)
		# the experiences are already on the same 'device' as RL model!

		# Compute the target Q value with Double Q-learning
		with torch.no_grad():
			argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
			max_q_next = self.q_target(s_next).gather(1,argmax_a)
			target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		# Mse regression
		q_loss = F.mse_loss(current_q_a, target_Q)
		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 40)
		self.q_net_optimizer.step()

		# Update the target net
		for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,steps):
		torch.save(self.q_net.state_dict(), "./model/{}k.pth".format(steps))

	def load(self,steps):
		self.q_net.load_state_dict(torch.load("./model/{}k.pth".format(steps)))
		self.q_target = copy.deepcopy(self.q_net)
		for p in self.q_target.parameters(): p.requires_grad = False


def evaluate_policy(env, model, turns = 3):
	scores = 0
	for j in range(turns):
		s, _ = env.reset()
		done, ep_r = False, 0
		while not done:
			# Take deterministic actions at test time
			a = model.select_action(s, deterministic=True)
			s_next, r, dw, tr, info = env.step(a)
			done = dw + tr
			ep_r += r
			s = s_next
		scores += ep_r
	return round(scores/turns, 1)


if __name__ == '__main__':
    main(opt)