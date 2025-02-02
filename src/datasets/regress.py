from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import scipy.io

import numpy as np
import torch

class CommCrime(Dataset):
	def __init__(self, seed=0, train=True):
		super().__init__()

		crime = scipy.io.loadmat('data/Crime.mat')

		X, group, y  = (crime['X'], crime['Scat'], crime['Y'])
		import pdb; pdb.set_trace()
		# reverse 0/1 group label
		# for general consistency and comparison to other results
		group = (~(group.astype(bool))).astype(group.dtype)

		X_train, X_test, g_train, g_test, y_train, y_test = train_test_split(X, group, y, 
					test_size=0.3, random_state=seed)
	
		if train:
			self.features = X_train
			self.attrs = g_train
			self.labels = y_train
		else:
			self.features = X_test
			self.attrs = g_test
			self.labels = y_test

		# import pdb; pdb.set_trace()

	def __getitem__(self, index):
		X = torch.from_numpy(self.features[index,:]).float()
		y = self.labels[index]
		attr = self.attrs[index]
		return X, torch.Tensor([y[0], int(attr)])

	def __len__(self):
		return len(self.labels)



