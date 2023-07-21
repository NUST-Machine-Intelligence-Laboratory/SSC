import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *




    
class Decoder(nn.Module): 
	def __init__(self):
		super(Decoder, self).__init__()

		self.main = []
		self.upsample_1 = nn.Sequential(
			# input: 1/16 * 1/16
			nn.ConvTranspose2d(256, 256, 4, 2, 1,  bias=False),
			nn.InstanceNorm2d(256),
			nn.ReLU(True),
			Conv2dBlock(256, 256, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),
			# input: 1/8 * 1/8
		)
		self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
			Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),
			# 1/4 * 1/4
		)
		self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
			Conv2dBlock(128, 64 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1/2 * 1/2
		)
		self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
			Conv2dBlock(64 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1 * 1
			nn.Conv2d(32, 3, 3, 1, 1),
			nn.Tanh())
		
		self.main += [Conv2dBlock(20, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
		self.main += [ResBlocks(1, 256, 'ln', 'relu', pad_type='zero')]
		#self.main += [self.upsample]
		
		self.main = nn.Sequential(*self.main)
		
	def forward(self, code):
		output = self.main(code)
		output = self.upsample_1(output)
		output = self.upsample_2(output)
		output = self.upsample_3(output)
		output = self.upsample_4(output)



		return output



