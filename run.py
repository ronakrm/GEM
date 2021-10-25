import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision

from torch_src.nn_utils import manual_seed, do_reg_epoch

from torch_src.datasets import MNISTF1
from torch_src.models.mnist import FFNetF1
from demd import DEMDFairLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
	manual_seed(args.train_seed)
	outString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.epochs)+'_lr_' + str(args.learning_rate)+'_wd_' + str(args.weight_decay)+'_bs_' + str(args.batch_size)+'_optim_' + str(args.optim)
	print(outString)
	
	model = FFNetF1()
	criterion = torch.nn.CrossEntropyLoss()
	# reg = DEMDFairLoss()
	print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

	optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

	train_dataset = MNISTF1('.data/', train=True)
	valid_dataset = MNISTF1('.data/', train=True)
	print(len(train_dataset))
	print(len(val_dataset))

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
											  shuffle=True, num_workers=1)

	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
											 shuffle=False, num_workers=1)

	for epoch in range(args.epochs):
		train_loss, train_accuracy = do_reg_epoch(model, train_loader, criterion, reg, epoch, args.epochs, optim=optim, device=device, outString=outString)

		with torch.no_grad():
			val_loss, val_accuracy = do_reg_epoch(model, val_loader, criterion, reg, epoch, args.epochs, optim=None, device=device, outString=outString)

		tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
				   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

		print('Saving model...')
		torch.save(model.state_dict(), outString + '.pt')

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser(description='Train a network')
	arg_parser.add_argument('--train_seed', type=int, default=0)
	arg_parser.add_argument('--dataset', type=str, default='cifar10')
	arg_parser.add_argument('--model', type=str, default='CIFAR10Net')
	arg_parser.add_argument('--batch_size', type=int, default=32)
	arg_parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
	arg_parser.add_argument('--epochs', type=int, default=20)
	arg_parser.add_argument('--n_classes', type=int, default=10)
	arg_parser.add_argument('--learning_rate', type=float, default=0.01)
	arg_parser.add_argument('--momentum', type=float, default=0.9)
	arg_parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay, or l2_regularization for SGD')
	args = arg_parser.parse_args()
	main(args)