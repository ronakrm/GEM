import torch
import numpy as np
from tqdm import tqdm

def getOutputs(model, dataloader, device):

	model.eval()

	acts = []
	labels = []

	for x, target in tqdm(dataloader, leave=False):
		(y_true, attr) = target[:, 1], target[:, 0].to(device)
		x, y_true = x.to(device), y_true.to(device).float()

		act = model(x).squeeze()

		acts.extend(act.detach().cpu())
		labels.extend(attr.detach().cpu())

	return torch.stack(acts), torch.stack(labels)

def getAcc(acts, labels):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		return ((y_sigs>0.5) == labels).float().mean().item()
	else:
		return (y_sig.max(1)[1] == labels).float().mean().item()

def getHist(acts, nbins=10):
	cdfs = torch.sigmoid(acts)
	dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
	return dist

def genClassificationReport(model, dataloader, dist, device):

	acts, labels = getOutputs(model, dataloader, device)

	groups = torch.unique(labels)

	hists = {}
	accs = {}

	for group in groups:
		gacts = acts[labels==group]
		glabels = labels[labels==group]
		accs[group] = getAcc(gacts, glabels)
		hists[group] = getHist(gacts)

	total_acc = getAcc(acts, labels)
	full_hist = getHist(acts)

	print('*'*5, 'Classification Report', '*'*5)
	print('Class\t\tAcc')
	for group in groups:
		print(f'{group}\t\t{accs[group]}')

	print(f'Total Acc: {total_acc}')

	print('Class\t\tHist')
	for group in groups:
		print(f'{group}\t\t{hists[group]}')
	print(f'Global Hist: {full_hist}')

	stacked = torch.stack(list(hists.values())).requires_grad_(requires_grad=True)
	demd = dist(stacked).item()
	print(f'Full dEMD Distance: {demd}')