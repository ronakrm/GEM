# data_utils.py

def getDatasets(name='binsizemnist', valid_also=True):

    if name == 'binsizemnist':
        from .datasets import BinarySizeMNIST

        train_dataset = BinarySizeMNIST(root='./data', train=True, download=True)
        if valid_also:
            valid_dataset = BinarySizeMNIST(root='./data', train=False, download=True)

    if valid_also:
        return train_dataset, valid_dataset
    else:
        return train_dataset