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

		# import pdb; pdb.set_trace()
		(y_true, attr) = target[:, 1], target[:, 0].to(device)
		x, y_true = x.to(device), y_true.to(device).float()#.unsqueeze(1)

		act = model(x).squeeze()
		y_sig = torch.sigmoid(act)
		recon_loss = criterion(y_sig, y_true)
		#loss = criterion(y_sigm, y_true.float())

		# import pdb; pdb.set_trace()

		reg_loss = reg(act, attr)

		loss = recon_loss + 0.001*reg_loss

		# for training
		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		nsamps += len(y_true)
		total_loss += loss.item()

		total_accuracy += ((y_sig>0.5) == y_true).float().mean().item()

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
