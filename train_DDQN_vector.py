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

parser.add_argument('--Max_train_steps', type=int, default=2E5, help='Max training steps')
parser.add_argument('--buffersize', type=int, default=2E5, help='Replay buffer size')
parser.add_argument('--save_interval', type=int, default=1E4, help='Model saving interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1E4, help='steps for random policy to explore')

parser.add_argument('--actor_envs', type=int, default=10, help='number of vectorized envs')
parser.add_argument('--init_explore_frac', type=float, default=1.0, help='init explore fraction')
parser.add_argument('--end_explore_frac', type=float, default=0.3, help='end explore fraction')
parser.add_argument('--decay_step', type=int, default=int(40e3), help='linear decay steps(total) for e-greedy noise')
parser.add_argument('--min_eps', type=float, default=0.02, help='minimal e-greedy noise')

parser.add_argument('--gamma', type=float, default=0.98, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
opt = parser.parse_args()
opt.state_dim = 5+27 # [dx,dy,orientation,v_linear,v_angular] + [lidar result]
opt.action_dim = 5

assert opt.actor_envs % 5 == 0
if opt.write: from torch.utils.tensorboard import SummaryWriter

def main(opt):
	# init DDQN model
	if not os.path.exists('model'): os.mkdir('model')
	model = DDQN_Agent(opt)
	if opt.Loadmodel: model.load(opt.ModelIdex)

	if opt.render: # render with Sparrow
		eval_envs = gym.vector.AsyncVectorEnv([lambda: gym.make('Sparrow-v0', dvc=device, np_state=True, render_mode='human') for _ in range(opt.actor_envs)])
		while True:
			ep_r, rate = evaluate_policy(eval_envs, model, deterministic=True)
			print('Score:', ep_r, ';  Win Rate:', rate, '\n')

	else: # trian with Sparrow
		if opt.write:
			# use SummaryWriter to record the training curve
			timenow = str(datetime.now())[0:-10]
			timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
			writepath = 'runs/Sparrow_v0' + timenow
			if os.path.exists(writepath): shutil.rmtree(writepath)
			writer = SummaryWriter(log_dir=writepath)

		# build vectorized environment and experience replay buffer
		buffer = ReplayBuffer(opt)
		envs = gym.vector.AsyncVectorEnv([lambda: gym.make('Sparrow-v0', dvc=device, np_state=True) for _ in range(opt.actor_envs)])

		s, info = envs.reset() # vectorized env has auto truncate mechanism, so we only reset() once.
		ep_r, total_steps = 0, 0

		while total_steps < opt.Max_train_steps:
			if total_steps < opt.random_steps:
				a = np.random.randint(0, opt.action_dim, opt.actor_envs)
			else:
				a = model.select_action(s, deterministic=False)
			s_next, r, dw, tr, info = envs.step(a) #dw(terminated): die or win; tr: truncated
			buffer.add(s, a, r, dw, tr)
			s = s_next
			total_steps += opt.actor_envs

			# log and record
			ep_r += r[0]
			if dw[0] or tr[0]:
				if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
				print('Vectorized Sparrow-v0:', 'steps: {}k'.format(round(total_steps / 1000,2)), 'score:', ep_r)
				ep_r = 0

			# train and fresh e-greedy noise
			if total_steps >= opt.random_steps:
				for _ in range(opt.actor_envs):
					model.train(buffer)
				# fresh vectorized e-greedy noise
				if total_steps % (10*opt.actor_envs) == 0:
					model.fresh_explore_prob(total_steps-opt.random_steps)

			# save model
			if (total_steps) % opt.save_interval == 0:
				model.save(int(total_steps/1e3))



class ReplayBuffer():
	'''Experience replay buffer(For vector env)'''
	def __init__(self, opt):
		self.device = device
		self.max_size = int(opt.buffersize/opt.actor_envs)
		self.state_dim = opt.state_dim
		self.actor_envs = opt.actor_envs
		self.ptr = 0
		self.size = 0
		self.full = False
		self.batch_size = opt.batch_size

		self.s = torch.zeros((self.max_size, opt.actor_envs, opt.state_dim))
		self.a = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.int64)
		self.r = torch.zeros((self.max_size, opt.actor_envs, 1))
		self.dw = torch.zeros((self.max_size, opt.actor_envs, 1), dtype=torch.bool)
		self.tr = torch.zeros((self.max_size, opt.actor_envs, 1),dtype=torch.bool)

	def add(self, s, a, r, dw, tr):
		'''add transitions to buffer'''
		self.s[self.ptr] = torch.from_numpy(s)
		self.a[self.ptr] = torch.from_numpy(a).unsqueeze(-1)  #[actor_envs,1]
		self.r[self.ptr] = torch.from_numpy(r).unsqueeze(-1)
		self.dw[self.ptr] = torch.from_numpy(dw).unsqueeze(-1)
		self.tr[self.ptr] = torch.from_numpy(tr).unsqueeze(-1)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		if self.size == self.max_size:
			self.full = True

	def sample(self):
		'''sample batch transitions'''
		if not self.full:
			ind = torch.randint(low=0, high=self.ptr - 1, size=(self.batch_size,))  # sample from [0, ptr-2]
		else:
			ind = torch.randint(low=0, high=self.size - 1, size=(self.batch_size,))  # sample from [0, size-2]
			if self.ptr - 1 in ind:
				ind = np.delete(ind, np.where(ind == (self.ptr - 1)))  # delate ptr - 1 in [0, size-2]

		env_ind = torch.randint(low=0, high=self.actor_envs, size=(len(ind),)) # [l,h)
		# [b, s_dim], #[b, 1], [b, 1], [b, s_dim], [b, 1], [b, 1]
		return (self.s[ind,env_ind,:].to(self.device), self.a[ind,env_ind,:].to(self.device),
				self.r[ind,env_ind,:].to(self.device), self.s[ind + 1,env_ind,:].to(self.device),
				self.dw[ind,env_ind,:].to(self.device), self.tr[ind, env_ind,:].to(self.device))

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
	def __init__(self,opt):
		self.q_net = Q_Net(opt.state_dim, opt.action_dim, opt.net_width).to(device)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.actor_envs = opt.actor_envs
		self.action_dim = opt.action_dim
		self.gamma = opt.gamma
		self.tau = 0.005

		# vectorized e-greedy exploration
		self.explore_frac_scheduler = LinearSchedule(opt.decay_step, opt.init_explore_frac, opt.end_explore_frac)
		self.p = torch.ones(opt.actor_envs)
		self.min_eps = opt.min_eps

	def fresh_explore_prob(self, steps):
		#fresh vectorized e-greedy noise
		explore_frac = self.explore_frac_scheduler.value(steps)
		i = int(explore_frac * self.actor_envs)
		explore = torch.arange(i) / (2 * i)  # 0 ~ 0.5
		self.p *= 0
		self.p[self.actor_envs - i:] = explore
		self.p += self.min_eps

	def select_action(self, s, deterministic):
		'''For envpool, input dim is [n,32], numpy.narray'''
		with torch.no_grad():
			s = torch.from_numpy(s).to(device)  # [n,32]
			a = self.q_net(s).argmax(dim=-1).cpu()  # [n]
			if deterministic:
				return a.numpy()
			else:
				replace = torch.rand(self.actor_envs) < self.p  # [n]
				rd_a = torch.randint(0, self.action_dim, (self.actor_envs,))
				a[replace] = rd_a[replace]
				return a.numpy()

	def train(self,replay_buffer):
		s, a, r, s_next, dw, tr = replay_buffer.sample()

		# Compute the target Q value with Double Q-learning
		with torch.no_grad():
			argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
			max_q_next = self.q_target(s_next).gather(1,argmax_a)
			target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		# Mse regression
		ct = ~tr # whether s and s_next are consistent(from one episode). If not, discard.
		if ct.all():
			q_loss = F.mse_loss(current_q_a, target_Q)
		else:
			# discard truncated s, because we didn't save its next state
			q_loss = torch.square(ct * (current_q_a - target_Q)).mean()
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


def evaluate_policy(envs, model, deterministic):
	s, info = envs.reset()
	num_env = s.shape[0]
	dones, total_r, total_rate = np.zeros(num_env, dtype=np.bool_), 0, 0
	while not dones.all():
		a = model.select_action(s, deterministic)
		s, r, dw, tr, info = envs.step(a)
		total_r += (~dones * r).sum()  # use last dones

		# rate of reaching target area
		rate = r * 0
		rate[r == 75] = 1
		total_rate += (~dones * rate).sum()  # use last dones

		dones += (dw + tr)
	return round(total_r / num_env, 1), round(total_rate / num_env, 3)


class LinearSchedule(object):
	def __init__(self, schedule_timesteps, initial_p, final_p):
		"""Linear interpolation between initial_p and final_p over"""
		self.schedule_timesteps = schedule_timesteps
		self.initial_p = initial_p
		self.final_p = final_p

	def value(self, t):
		fraction = min(float(t) / self.schedule_timesteps, 1.0)
		return self.initial_p + fraction * (self.final_p - self.initial_p)

if __name__ == '__main__':
    main(opt)