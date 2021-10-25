import torch
import numpy as np
import random
from tqdm import tqdm

def do_epoch(model, dataloader, criterion, epoch, nepochs, optim=None, device='cpu', outString=''):
	# saves last two epochs gradients for computing finite difference Hessian
	total_loss = 0
	total_accuracy = 0
	nsamps = 0
	if optim is not None:
		model.train()
	else:
		model.eval()

	for x, y_true in tqdm(dataloader, leave=False):
	#for _, (x, y_true) in enumerate(dataloader):
		x, y_true = x.to(device), y_true.to(device)
		y_pred = model(x)
		loss = criterion(y_pred, y_true)
		#loss = criterion(y_pred, y_true.float())

		# for training
		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		nsamps += len(y_true)
		total_loss += loss.item()
		total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()

	mean_loss = total_loss / len(dataloader)
	mean_accuracy = total_accuracy / len(dataloader)

	return mean_loss, mean_accuracy


def do_reg_epoch(model, dataloader, criterion, reg, epoch, nepochs, optim=None, device='cpu', outString=''):
	# saves last two epochs gradients for computing finite difference Hessian
	total_loss = 0
	total_accuracy = 0
	nsamps = 0
	if optim is not None:
		model.train()
	else:
		model.eval()

	for x, target in tqdm(dataloader, leave=False):
	#for _, (x, y_true) in enumerate(dataloader):
		(y_true, attr) = target[0], target[1]
		x, y_true = x.to(device), y_true.to(device)
		y_pred = model(x)
		loss = criterion(y_pred, y_true)
		#loss = criterion(y_sigm, y_true.float())

		# reg = reg(y_sigm, attr)
		reg = 0

		total_loss = loss + 0.01*reg

		# for training
		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		nsamps += len(y_true)
		total_loss += loss.item()
		total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()


	mean_loss = total_loss / len(dataloader)
	mean_accuracy = total_accuracy / len(dataloader)

	return mean_loss, mean_accuracy


def manual_seed(seed):
	print("Setting seeds to: ", seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
