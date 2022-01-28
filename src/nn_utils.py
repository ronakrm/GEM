import torch
import numpy as np
import random
from tqdm import tqdm

from src.utils import genClassificationReport

def do_reg_epoch(model, dataloader, criterion, reg, dist, 
					epoch, nepochs, lambda_reg, nbins,
					optim=None, device='cpu', outString=''):
	# saves last two epochs gradients for computing finite difference Hessian
	total_loss = 0
	total_accuracy = 0
	nsamps = 0
	if optim is not None:
		model.train()
		reg.train()
	else:
		model.eval()
		reg.eval()

	acts = []
	targets = []
	attrs = []

	for x, target in tqdm(dataloader):

		(y_true, attr) = target[:, 0], target[:, 1].to(device)
		x, y_true = x.to(device), y_true.to(device).float()#.unsqueeze(1)

		act = model(x).squeeze()
		y_sig = torch.sigmoid(act)
		recon_loss = criterion(y_sig, y_true)

		reg_loss = reg(act, attr)
		# reg_loss = reg(X=None, y=y_true, out=act, sensitive=attr)

		loss = recon_loss + lambda_reg*reg_loss

		# for training
		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		nsamps += len(y_true)
		total_loss += loss.item()

		total_accuracy += ((y_sig>0.5) == y_true).float().mean().item()

		acts.extend(act.detach().cpu())
		targets.extend(y_true.detach().cpu())
		attrs.extend(attr.detach().cpu())

	mean_loss = total_loss / len(dataloader)
	mean_accuracy = total_accuracy / len(dataloader)

	tacts = torch.stack(acts)
	ttargets = torch.stack(targets)
	tattrs = torch.stack(attrs)

	valid_dist = reg(tacts, tattrs)

	if optim is None:
		accs, dp, eo, demd = genClassificationReport(tacts, ttargets, tattrs, dist=dist, nbins=nbins)
		vals = {}
		vals['maxacc'] = max(accs.values()).item()
		vals['minacc'] = min(accs.values()).item()
		vals['dp_gap'] = (max(dp.values()) - min(dp.values())).item()
		vals['eo_gap'] = (max(eo.values()) - min(eo.values())).item()
	else:
		vals = None

	return mean_loss, mean_accuracy, valid_dist, vals

