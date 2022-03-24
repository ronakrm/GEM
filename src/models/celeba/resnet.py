import torch

def resnet18(num_classes=2, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=num_classes)
    return model

def resnet50(num_classes=2, **kwargs):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False, num_classes=num_classes)
    return model