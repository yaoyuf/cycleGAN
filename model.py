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
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
            
print('initialize network with %s' % init_type)
net.apply(init_func)          

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class ResnetBlock(nn.Module):
    """Define a Resnet block, we use padding type: reflect"""

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)
        

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
            
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)
    
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# define netowrks
def Generator(input_nc, output_nc, n_filter, norm='batch', dropout=False, init_type='normal', init_gain=0.02, use_cuda=False):
	net = None
    	if norm == 'batch':
        	norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True) 
       		 net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
   	init.normal_(m.weight.data, 0.0, init_gain=0.02)
	return init_net(net, init_type, init_gain, gpu_ids)


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

		self.loss_names = ['genLoss_A', 'disLoss_A', 'cycleLoss_A', 'genLoss_B', 'disLoss_B', 'cycleLoss_B']
		self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
		
		# define generator network
		self.gen_A = Generator(args.input_nc, args.output_nc, args.n_filter, args.norm, args.dropout, args.init_type, args.init_gain, args.is_gpu)
		self.gen_B = Generator(args.output_nc, args.input_nc, args.n_filter, args.norm, args.dropout, args.init_type, args.init_gain, args.is_gpu)
		
		if self.is_train:
			self.model_names = ['gen_A', 'dis_A', 'gen_B', 'dis_B']
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
			self.model_names = ['gen_A', 'gen_B']

	def lr_update(self):
		for lr in self.lr:
			lr.step()
		lr = self.optimizer_G.param_groups[0]['lr']
		print('learning rate %.7f' % (lr))

	def model_save(self,epoch):
		for net in self.model_names:
			file = '%s_%s.pth' % (epoch, net)
			path = os.path.join(self.save_dir, file)
			model = getattr(self, net)
			if self.args.is_gpu and torch.cuda.is_available():
				torch.save(model.module.cpu().state_dict(), path)
				model.cuda()
			else:
				torch.save(model.cpu().state_dict(), path)

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
		
		self.genLoss_A = self.ganLoss(self.dis_A(self.fake_B), True)
		self.genLoss_B = self.ganLoss(self.dis_B(self.fake_A), True)
		self.cycleLoss_A = self.cycleLoss(self.rec_A, self.real_A) * self.args.lambda_A
		self.cycleLoss_B = self.cycleLoss(self.rec_B, self.real_B) * self.args.lambda_B
		self.loss_G = self.genLoss_A + self.genLoss_B + self.cycleLoss_A + self.cycleLoss_B
		self.loss_G.backward()

		self.optimizer_G.step()

	def backward_D(self):
		self.optimizer_D.zero_grad()

		fake_B = self.fake_B
		loss_real_A = self.ganLoss(self.dis_A(self.real_B), True)
		loss_fake_A = self.ganLoss(self.dis_A(fake_B.detach()), False)
		loss_A = (loss_real_A + loss_fake_A) * 0.5
		loss_A.backward()
		self.disLoss_A = loss_A

		fake_A = self.fake_A
		loss_real_B = self.ganLoss(self.dis_B(self.real_B), True)
		loss_fake_B = self.ganLoss(self.dis_B(fake_A.detach()), False)
		loss_B = (loss_real_B + loss_fake_B) * 0.5
		loss_B.backward()
		self.disLoss_B = loss_B
		
		self.optimizer_D.step()


	def Optimize(self, input_A, input_B):

		self.forward(input_A, input_B)

		self.set_required_grad([self.dis_A, self.dis_B], False)
		self.backward_G()
		self.set_required_grad([self.dis_A, self.dis_B], True)
		self.backward_D()
