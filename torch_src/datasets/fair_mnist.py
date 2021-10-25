

import numpy as np
import torch.utils.data
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip


from torchvision.datasets import MNIST


from __future__ import print_function
import os
import os.path
import numpy as np
import pandas as pd
import sys
import torch
import torch.utils.data as data
from PIL import Image


class FairMNIST(data.Dataset):
	"""

	"""
	def __init__(self, dataroot='./data/fairmnist/', attr_data_file="fairmnist.csv", train=True):

		super(FairMNIST, self).__init__()

		self.dataroot = dataroot
		self.attr_data_file = attr_data_file

		try:
			if train:
				self.attr_data = pd.read_csv(os.path.join(self.dataroot, '_train_', self.attr_data_file), low_memory=False)
			else:
				self.attr_data = pd.read_csv(os.path.join(self.dataroot, '_valid_', self.attr_data_file), low_memory=False)
		except FileNotFoundError:
			raise ValueError("Image attribute file {:s} does not exist".format(os.path.join(dataroot, self.attr_data_file)))
		except e:
			print(e)
		
		self.n_samples = len(self.attr_data)
		print("N samples: ", self.n_samples)


	def __getitem__(self, index):

		imgrow = self.attr_data.iloc[index]
		pilimg = Image.open(os.path.join(self.dataroot, imgrow['FilePath']))
		im = np.array(pilimg)
		return im, int(imgrow['Target']), int(imgrow['Attribute'])


	def __len__(self):
		return self.n_samples


def randomResize(img):

	

# Generate Fair MNIST
if __name__ == "__main__":



train_transformation = Compose([
	ToTensor(),
	Normalize((0.1307,), (0.3081,)),
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transformation)
val_dataset = MNIST(root='./data', train=False, download=True, transform=train_transformation)


