import numpy as np
import torch.utils.data
# from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip

def getDatasets(name='adult',
					target=None,
					attr_cols='race',
					data_augment=False,
					download=True):

	if name == 'adult':
		from src.datasets import Adult
		train_dataset = Adult(attr_col=attr_cols, train=True)
		valid_dataset = Adult(attr_col=attr_cols, train=False)
	elif name == 'acs-employ':
		from src.datasets import ACSEmployment
		train_dataset = ACSEmployment(train=True)
		valid_dataset = ACSEmployment(train=False)
	elif name == 'acs-income':
		from src.datasets import ACSIncome
		train_dataset = ACSIncome(train=True)
		valid_dataset = ACSIncome(train=False)


	elif name == 'binary-mnist':
		from src.datasets import BinarySizeMNIST

		train_transformation = Compose([
			ToTensor(),
			Normalize((0.1307,), (0.3081,)),
		])

		train_dataset = BinarySizeMNIST(root='./data', train=True, download=download, transform=train_transformation)
		# train_dataset = LabelSubsetWrapper(train_dataset, which_labels=(0,1))
		
		valid_dataset = BinarySizeMNIST(root='./data', train=False, download=download, transform=train_transformation)
		# val_dataset = LabelSubsetWrapper(val_dataset, which_labels=(0,1))

	elif name == 'celeba-test':
		target = 'Smiling'
		attr_cols = ['Young']
		# attr_cols = ['Young', 'Brown_Hair']
		from src.datasets import CelebA

		train_dataset = CelebA('./data/celeba/', train=True, target=target, 
										spurious=attr_cols,
										n_samples=1000)
		valid_dataset = CelebA('./data/celeba/', train=False, target=target, 
										spurious=attr_cols,
										n_samples=1000)
	
	elif name == 'celeba':
		target = 'Smiling'
		attr_cols = ['Young']
		from src.datasets import CelebA

		train_dataset = CelebA(root='./data/celeba/', train=True, target=target, 
										spurious=attr_cols)
		valid_dataset = CelebA('./data/celeba/', train=False, target=target, 
										spurious=attr_cols)

	# elif name == 'mnist':
	# 	from torchvision.datasets import MNIST

	# 	train_transformation = Compose([
	# 		ToTensor(),
	# 		Normalize((0.1307,), (0.3081,)),
	# 	])

	# 	train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transformation)
	# 	val_dataset = MNIST(root='./data', train=False, download=True, transform=train_transformation)


	# elif name == 'cifar10':
	# 	from torchvision.datasets import CIFAR10

	# 	if data_augment:
	# 		print("Doing data augmentation")
	# 		train_transformation = Compose(
	# 			[
	# 				RandomCrop(32, padding=4),
	# 				RandomHorizontalFlip(),
	# 				ToTensor(),
	# 				Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# 			]
	# 		)
	# 	else:
	# 		train_transformation = Compose([
	# 			ToTensor(),
	# 			Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# 		])

	# 	val_transformation = Compose([
	# 			ToTensor(),
	# 			Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# 		])
		
	# 	train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transformation)
		
	# 	val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transformation)

	# elif name == 'celeba_ID':
	# 	from dataset import OurCelebA

	# 	train_dataset = OurCelebA('./data', n_classes=100)

	# 	valid_dataset = OurCelebA('./data', n_classes=100)

	else:
		error('unknown dataset')

	return train_dataset, valid_dataset
