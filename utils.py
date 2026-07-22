import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def CosineWithWarmup(optimizer, warmup=1000, total_steps=2000):
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, 0.01, 1.0, warmup),
            CosineAnnealingLR(optimizer, total_steps-warmup, 1e-6)
        ],
        milestones=[warmup])

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class ImageNet32(torch.utils.data.Dataset):
    def __init__(self, root, split, transform=None):
        self.path = f'{root}{split}_images.npy'
        self.images = None  # opened lazily: a memmap must not be pickled into workers
        self.labels = np.load(f'{root}{split}_labels.npy')
        self.transform = transform

    def __getstate__(self):
        return {**self.__dict__, 'images': None}  # each worker re-opens its own memmap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode='r')  # N x 32 x 32 x 3
        img = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1)  # C H W uint8, copied out of the memmap
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])
