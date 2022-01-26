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

def getDP(acts, labels):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		return ((y_sigs>0.5) == True).float().mean()

def getEO(acts, labels):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		a = ((y_sigs>0.5) == True).bool()
		b = a & (labels==1).bool()
		return (b).float().mean()


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

def genClassificationReport(acts, targets, attrs, dist=None, nbins=10):

	groups = torch.unique(attrs).numpy()

	hists = {}
	accs = {}
	dp = {}
	eo = {}

	for group in groups:
		gacts = acts[attrs==group]
		gtargets = targets[attrs==group]
		accs[group] = getAcc(gacts, gtargets).detach().cpu().numpy()
		dp[group] = getDP(gacts, gtargets).detach().cpu().numpy()
		eo[group] = getEO(gacts, gtargets).detach().cpu().numpy()
		hists[group] = getHist(gacts, nbins=nbins)

	total_acc = getAcc(acts, targets)
	full_hist = getHist(acts, nbins=nbins).detach().cpu().numpy()

	print('*'*5, 'Classification Report', '*'*5)
	with np.printoptions(precision=3, suppress=True):
		print('Class\t\tAcc\t\tDP\t\tEO\t\tHist')
		for group in groups:
			print(f'{group}\t\t{accs[group]:.4f}\t\t{dp[group]:.4f}\t\t{eo[group]:.4f}\t\t{hists[group].detach().cpu().numpy()}')
		print(f'Total Acc: {total_acc}')
		print(f'Global Hist: {full_hist}')

	if dist is not None:
		stacked = torch.stack(list(hists.values()))
		demd = dist(stacked).item()
		print(f'Full dEMD Distance: {demd}')

	return accs, dp, eo, demd