import torch
import numpy as np
from tqdm import tqdm

def getOutputs(model, dataloader, device):

	model.eval()

	acts = []
	targets = []
	attrs = []

	for x, target in tqdm(dataloader, leave=False):
		(y_true, attr) = target[:, 1], target[:, 0].to(device)
		x, y_true = x.to(device), y_true.to(device).float()

		act = model(x).squeeze()

		acts.extend(act.detach().cpu())
		targets.extend(y_true.detach().cpu())
		attrs.extend(attr.detach().cpu())

	return torch.stack(acts), torch.stack(targets), torch.stack(attrs)

def getAcc(acts, labels):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		return ((y_sigs>0.5) == labels).float().mean()
	else:
		return (y_sig.max(1)[1] == labels).float().mean()

def getHist(acts, nbins=10):
	cdfs = torch.sigmoid(acts)
	dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
	return dist/sum(dist)

def genClassificationReport(acts, targets, attrs, dist=None):

	groups = torch.unique(attrs).numpy()

	hists = {}
	accs = {}

	for group in groups:
		gacts = acts[attrs==group]
		gtargets = targets[attrs==group]
		accs[group] = getAcc(gacts, gtargets).detach().cpu().numpy()
		hists[group] = getHist(gacts)

	total_acc = getAcc(acts, targets)
	full_hist = getHist(acts).detach().cpu().numpy()

	print('*'*5, 'Classification Report', '*'*5)
	with np.printoptions(precision=3, suppress=True):
		print('Class\t\tAcc\t\tHist')
		for group in groups:
			print(f'{group}\t\t{accs[group]}\t\t{hists[group].detach().cpu().numpy()}')
		print(f'Total Acc: {total_acc}')
		print(f'Global Hist: {full_hist}')

	if dist is not None:
		stacked = torch.stack(list(hists.values())).requires_grad_(requires_grad=True)
		demd = dist(stacked).item()
		print(f'Full dEMD Distance: {demd}')