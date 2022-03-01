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

def getDP(acts, labels, threshold=0.5):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		return ((y_sigs>threshold) == True).float().mean()

def getEO(acts, labels, threshold=0.5):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		a = ((y_sigs>threshold) == True).bool()
		b = a & (labels==1).bool()
		return (b).float().mean()


def getAcc(acts, labels, threshold=0.5):
	y_sigs = torch.sigmoid(acts)

	if len(y_sigs.shape) == 1:
		return ((y_sigs>threshold) == labels).float().mean()
	else:
		return (y_sig.max(1)[1] == labels).float().mean()

def getHist(acts, nbins=10):
	cdfs = torch.sigmoid(acts)
	dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
	return dist/sum(dist)

def getGTDist(targets, uniq_targs):

	dist = []
	for t in uniq_targs:
		d = sum(targets==t)/len(targets)
		dist.append(d)

	return dist

def genClassificationReport(acts, targets, attrs, dist=None, nbins=10, threshold=0.5, verbose=False):

	groups = torch.unique(attrs).numpy()
	targs = torch.unique(targets)

	gtdists = {}
	hists = {}
	accs = {}
	dp = {}
	eo = {}

	for group in groups:
		gacts = acts[attrs==group]
		gtargets = targets[attrs==group]
		gtdists[group] = getGTDist(gtargets, targs)
		accs[group] = getAcc(gacts, gtargets, threshold=threshold).detach().cpu().numpy()
		dp[group] = getDP(gacts, gtargets, threshold=threshold).detach().cpu().numpy()
		eo[group] = getEO(gacts, gtargets, threshold=threshold).detach().cpu().numpy()
		hists[group] = getHist(gacts, nbins=nbins)

	total_gt = getGTDist(targets, targs)
	total_acc = getAcc(acts, targets, threshold=threshold)
	total_dp = getDP(acts, targets, threshold=threshold).detach().cpu().numpy()
	total_eo = getEO(acts, targets, threshold=threshold).detach().cpu().numpy()
	full_hist = getHist(acts, nbins=nbins).detach().cpu().numpy()

	if verbose:
		print('*'*5, 'Classification Report', '*'*5)
		with np.printoptions(precision=3, suppress=True):
			print('Class\t\tTruth\t\t\tAcc\t\tDP\t\tEO\t\tHist')
			for group in groups:
				print(f'{group}\t\t{gtdists[group][0]:.4f} {gtdists[group][1]:.4f}\t\t{accs[group]:.4f}\t\t{dp[group]:.4f}\t\t{eo[group]:.4f}\t\t{hists[group].detach().cpu().numpy()}')

			print(f'Total\t\t{total_gt[0]:.4f} {total_gt[1]:.4f}\t\t{total_acc:.4f}\t\t{total_dp:.4f}\t\t{total_eo:.4f}\t\t{full_hist}')

	if dist is not None:
		stacked = torch.stack(list(hists.values()))
		demd = dist(stacked).item()
		if verbose:
			print(f'Full dEMD Distance: {demd}')

	return total_gt, accs, dp, eo, demd