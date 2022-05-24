import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .clean import clean_dataset

class GerryDataset(Dataset):
	def __init__(self, root= 'data/dataset/', dataset='communities', seed=0, train=True):
			super().__init__()

			# communities_dataset = "./dataset/communities.csv"
			# communities_attributes = "./dataset/communities_protected.csv"
			# lawschool_dataset = "./dataset/lawschool.csv"
			# lawschool_attributes = "./dataset/lawschool_protected.csv"
			# adult_dataset = "./dataset/adult.csv"
			# adult_attributes = "./dataset/adult_protected.csv"
			# student_dataset = "./dataset/student-mat.csv"
			# student_attributes = "./dataset/student_protected.csv"

			dset = root + dataset + '.csv'
			dattr = root + dataset + '_protected.csv'


			self.features, self.group, self.label = clean_dataset(dset, dattr, centered=False)

			X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
	    		self.features, self.label, self.group, test_size=0.2, random_state=seed)
		
			if train:
				X = X_train
				y = y_train
				group = group_train
			else:
				X = X_test
				y = y_test
				group = group_test

			scalar = StandardScaler()
			self.X = scalar.fit_transform(X)

			### encode all possible subgroups!
			self.attrs = []
			for i in range(group.shape[0]):
				subgroup = int(''.join(group.iloc[i,:].astype('str').values.tolist()),2)
				self.attrs.append(subgroup)
			
			self.y = y.values


	def __getitem__(self, index):
		X = torch.from_numpy(self.X[index, :]).float()
		y = self.y[index]
		attr = self.attrs[index]
		return X, torch.Tensor([int(y), int(attr)])

	def __len__(self):
		return len(self.y)