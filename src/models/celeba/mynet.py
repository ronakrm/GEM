from torch import nn

class CelebANet(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.feature_extractor = nn.Sequential(
			nn.Conv2d(3, 20, kernel_size=3),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(20, 30, kernel_size=3),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(30, 40, kernel_size=3),
			nn.MaxPool2d(2),
			nn.ReLU(),
			nn.Conv2d(40, 50, kernel_size=3),
			nn.MaxPool2d(2),
			nn.Dropout2d(),
		)
		
		self.classifier = nn.Sequential(
			nn.Linear(4950, 1000),
			nn.ReLU(),
			nn.Linear(1000, 100),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(100, 1),
		)

	def forward(self, x):
		features = self.feature_extractor(x)
		features = features.view(x.shape[0], -1)
		logits = self.classifier(features)
		return logits
