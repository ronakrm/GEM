import torch.nn as nn


class ACSNet(nn.Module):
    def __init__(self, input_size, hidden_size=10, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out1 = self.fc2(out)
        return out1

class ACSRegressor(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
     
    def forward(self, x):
        out = self.fc1(x)
        return out