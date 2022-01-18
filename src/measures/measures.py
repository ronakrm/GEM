import torch.nn


class Measure(torch.nn.Module):

	def __init__(self):
		self.super.__init__()

	def forward(pred, true, attr):
		return NotImplementedError

class DemographicParity(Measure):
	def __init__(self):
		self.super.__init__()

	def forward(pred, true, attr):

		
