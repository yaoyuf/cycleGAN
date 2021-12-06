import os
import torch
import torch.nn as nn
import itertools
from torch.optim import lr_scheduler

'''
Model Initialization
'''
def init_model(model, init_type='normal', init_gain=0.02, use_cuda=False):
	if use_cuda:
		model.to('cuda')
	
	def weights_init(module):
		name = m.__class__.__name__
		if name.find('Conv') != -1:
			nn.init.normal_(module.weight.data, 0.0, init_gain)
			if hasattr(module, 'bias') and module.bias is not None:
				nn.init.constant_(m.bias.data, 0.0)
		elif name.find('BatchNorm') != -1:
			nn.init.normal_(module.weight.data, 1.0, init_gain)
			nn.init.constant_(module.bias.data, 0.0)
	model.apply(weights_init)

	return model

# define netowrks
def Generator(input_nc, output_nc, n_filter, norm='batch', dropout=False, init_type='normal', init_gain=0.02, use_cuda=False):

	return


'''
Discriminator Network
'''
class PatchGANDiscriminator(nn.Module):
	# Using Patch GAN as Discriminator 
	def __int__(self, input_nc, n_filter, n_layers=3, norm_layer=nn.BatchNorm2d):
		super(PatchGANDiscriminator, self).__init__()
		
		bias = norm_layer != nn.BatchNorm2d
		kernel = 4
		padding = 1
		layers = [nn.Conv2d(input_nc, n_filter, kernel_size=kernel, stride=2, padding=padding), nn.LeakyReLU(0.2,True)]
		k = 1
		for i in range(n_layers):
			layers += [
				nn.Conv2d(n_filter*k, n_filter*k*2, kernel_size=kernel, stride=2, padding=padding, bias=bias),
				norm_layer(n_filter*k*2),
				nn.LeakyReLU(0.2,True)
			]
			k *= 2
		layers += [
			nn.Conv2d(n_filter*k, n_filter*k*2, kernel_size=kernel, stride=1, padding=padding, bias=bias),
			norm_layer(n_filter*k*2),
			nn.LeakyReLU(0.2,True),
			nn.Conv2d(n_filter*k*2, 1 , kernel_size=kernel, stride=1, padding=padding, bias=bias)
		]
		
		self.model = nn.Sequential(*layer)
	
	def forward(self, input):
		return self.model(input)

def Discriminator(input_nc, n_filter, n_layers=3, norm='batch', init_type='normal', init_gain=0.02, use_cuda=False):
	model = PatchGANDiscriminator(input_nc, n_filter, n_layers)
	return init_model(model, init_type, init_gain, use_cuda)


'''
Adversarial Loss
'''
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


'''
Cycle GAN model
'''
class CycleGAN():

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
			if args.lr_policy == 'linear':
				def lambda_rule(epoch):
					lr_l = 1.0 - max(0, epoch + args.epoch_count - args.n_epochs) / float(args.n_epochs_decay + 1)
					return lr_l		
				self.lr = [lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule), lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)]
			else:
				self.lr = [lr_scheduler.StepLR(self.optimizer_G, step_size=args.lr_decay_iters, gamma=0.1),lr_scheduler.StepLR(self.optimizer_D, step_size=args.lr_decay_iters, gamma=0.1)]		
		else:
			self.model_names = ['G_A', 'G_B']

	def lr_update(self):
		for lr in self.lr:
			lr.step()
		lr = self.optimizer_G.param_groups[0]['lr']
		print('learning rate %.7f' % (lr))

	def set_required_grad(self, network, requires_grad):
		for net in network:
			for param in net.parameters():
				param.requires_grad = requires_grad

	def forward(self, input_A, input_B):
		# compute fake and rec pictures using generator
		self.real_A = input_A
		self.real_B = input_B
		self.fake_B = self.gen_A(self.real_A)
		self.fake_A = self.gen_B(self.real_B)
		self.rec_A = self.gen_B(self.fake_B)
		self.rec_B = self.gen_A(self.fake_A)
	
	def backward_G(self):
		self.optimizer_G.zero_grad()
		
		self.ganLoss_A = self.ganLoss(self.dis_A(self.fake_B), True)
		self.ganLoss_B = self.ganLoss(self.dis_B(self.fake_A), True)
		self.cycleLoss_A = self.cycleLoss(self.rec_A, self.real_A) * self.args.lambda_A
		self.cycleLoss_B = self.cycleLoss(self.rec_B, self.real_B) * self.args.lambda_B
		self.loss_G = self.ganLoss_A + self.ganLoss_B + self.cycleLoss_A + self.cycleLoss_B
		self.loss_G.backward()

		self.optimizer_G.step()

	def backward_D(self):
		self.optimizer_D.zero_grad()

		fake_B = self.fake_B
		loss_real_A = self.ganLoss(self.dis_A(self.real_B), True)
		loss_fake_A = self.ganLoss(self.dis_A(fake_B.detach()), False)
		loss_A = (loss_real_A + loss_fake_A) * 0.5
		loss_A.backward()

		fake_A = self.fake_A
		loss_real_B = self.ganLoss(self.dis_B(self.real_B), True)
		loss_fake_B = self.ganLoss(self.dis_B(fake_A.detach()), False)
		loss_B = (loss_real_B + loss_fake_B) * 0.5
		loss_B.backward()
		
		self.optimizer_D.step()


	def Optimize(self, input_A, input_B):

		self.forward(input_A, input_B)

		self.set_required_grad([self.dis_A, self.dis_B], False)
		self.backward_G()
		self.set_required_grad([self.dis_A, self.dis_B], True)
		self.backward_D()
