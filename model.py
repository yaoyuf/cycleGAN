import os
from typing import Generator
import torch
import itertools

# define netowrks
def Generator(input_nc, output_nc, n_filter, norm='batch', dropout=False, init_type='normal', init_gain=0.02, use_cuda=False):

	return

def Discriminator(input_nc, n_filter, n_layers=3, norm='batch', init_type='normal', init_gain=0.02, use_cuda=False):

	return

class GANLoss(nn.Module):
	# GANLoss calculator
	def __init__(self):
		super(GANLoss, self).__init__()
		self.real_label = torch.tensor(1.0)
		self.fake_label = torch.tensor(0.0)
		self.loss = nn.MSELoss()
	
	def __call__(self, prediction, goal):
		if goal:
			target = self.real_label.expand_as(prediction)
		else:
			target = self.fake_label.expand_as(prediction)
		loss = self.loss(prediction, target)

class CycleGAN():
	# CycleGan model
	def GANLoss():

		return

	def __init_(self, args):
		super(CycleGAN, self).__init__()
		self.args = args
		self.is_train = args.is_train
		self.device = torch.device('cuda') if args.is_gpu else torch.device('cpu')
		self.save_dir = os.path.join(args.checkpoints_dir, args.name)

		self.loss_names = ['G_A', 'D_A', 'cycle_A', 'G_B', 'D_B', 'cycle_B']
		self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
		
		# define generator network
		self.gen_A = Generator(args.input_nc, args.output_nc, args.n_filter, args.norm, args.dropout, args.init_type, args.init_gain, args.is_gpu)
		self.gen_B = Generator(args.output_nc, args.input_nc, args.n_filter, args.norm, args.dropout, args.init_type, args.init_gain, args.is_gpu)
		
		if self.is_train:
			self.model_names = ['G_A', 'D_A', 'G_B', 'D_B']
			# define discriminator network
			self.dis_A = Discriminator(args.output_nc, args.n_filter, args.n_layers, args.norm, args.init_type, args.init_gain, args.is_gpu)
			self.dis_B = Discriminator(args.input_nc, args.n_filter, args.n_layers, args.norm, args.init_type, args.init_gain, args.is_gpu)
			# define loss
			self.ganLoss = GANLoss().to(self.device)
			self.cycleLoss = torch.nn.L1loss()
			# define optimizer
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen_A.parameters(), self.gen_B.parameters()), lr=args.lr, betas=(args.beta, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()), lr=args.lr, betas=(args.beta, 0.999))
		else:
			self.model_names = ['G_A', 'G_B']
		
	def forward(self, input_A, input_B):
		# compute fake and rec pictures using generator
		self.real_A = input_A
		self.real_B = input_B
		self.fake_B = self.gen_A(self.real_A)
		self.fake_A = self.gen_B(self.real_B)
		self.rec_A = self.gen_B(self.fake_B)
		self.rec_B = self.gen_A(self.fake_A)
	
	def backward_G():

	def backward_D():
		

	def Optimize(self, input_A, input_B):

		self.forward(input_A, input_B)
		self.backward_G()
		self.backward_D()
