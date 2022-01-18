import numpy as np
import torch.utils.data
# from torch.utils.data import Subset
# from torchvision.transforms import Compose, ToTensor, Normalize, 
# from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip

def getDatasets(name='adult',
					attr_col='race',
					data_augment=False,
					download=True):

	if name == 'adult':
		from torch_src.datasets import Adult
		train_dataset = Adult(attr_col=attr_col, train=True)
		valid_dataset = Adult(attr_col=attr_col, train=False)

	# elif name == 'binary-mnist':
	# 	from torchvision.datasets import MNIST

	# 	train_transformation = Compose([
	# 		ToTensor(),
	# 		Normalize((0.1307,), (0.3081,)),
	# 	])

	# 	train_dataset = MNIST(root='./data', train=True, download=True, transform=train_transformation)
	# 	train_dataset = LabelSubsetWrapper(train_dataset, which_labels=(0,1))
		
	# 	val_dataset = MNIST(root='./data', train=False, download=True, transform=train_transformation)
	# 	val_dataset = LabelSubsetWrapper(val_dataset, which_labels=(0,1))

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
