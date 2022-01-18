from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
import torch

import h5py

class BinarySizeMNIST(MNIST):
    SIZE = 64
    SMALL_START = 18
    SMALL_END = 46

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(BinarySizeMNIST, self).__init__(root, train, transform, target_transform,
                                      download)
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
            bindatafile = root + '/' + type(self).__name__ + '/train.h5'
        else:
            data_file = self.test_file
            bindatafile = root + '/' + type(self).__name__ + '/valid.h5'
        
        # self.data, self.targets = torch.load(
            # Path(self.root, self.processed_folder, data_file))

        if Path(bindatafile).exists():
            try:
                myh5 = h5py.File(bindatafile, 'r')
                self.data = np.array(myh5['data'])
                self.targets = np.array(myh5['targets'])
                self.attrs = np.array(myh5['attrs'])

                print('Loaded dataset...')
            except:
                print('*'*70)
                print('Possibly Corrupted Data, delete files and rerun to regenerate.')
                print('*'*70)

        else:
            self._build()
            hf = h5py.File(bindatafile, 'w')
            hf.create_dataset('data', data=self.data)
            hf.create_dataset('targets', data=self.targets)
            hf.create_dataset('attrs', data=self.attrs)
            hf.close()

    def _build(self):
        data_, targets_, attrs_ = [], [], []

        print("Building dataset...")
        for [img, target] in tqdm(zip(self.data, self.targets)):
            img = img.numpy()

            # We create "small" and
            small = np.zeros([self.SIZE, self.SIZE])
            small[self.SMALL_START:self.SMALL_END, self.SMALL_START:self.
                  SMALL_END] += img
            data_.append(small)
            targets_.append(target)
            attrs_.append(0)

            # "big" versions of each image in the MNIST dataset
            big = np.array(Image.fromarray(img).resize((self.SIZE, self.SIZE)))

            data_.append(big)
            targets_.append(target)
            attrs_.append(1)

        self.data = data_
        self.targets = targets_
        self.attrs = attrs_

    def __getitem__(self, index):
        img, target, attr = self.data[index], self.targets[index], self.attrs[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # (class, big_or_small, index)
        # switched: predict size (first) attr, target is attr
        target = torch.tensor([int(attr), int(target), index])

        return img, target