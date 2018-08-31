import torch
import numpy as np
from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor
from model.image_folder_with_path import ImageFolderWithPaths, remove_none_collate
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

"""
https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           normalize=None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # # define transforms
    # valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    # if augment:
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])

    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     nomalize,
    # ])

    # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
    # copy transformation for vgg here

    if (normalize is not None):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        val_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])



    # # load the dataset
    # train_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=train_transform,
    # )
    #
    # valid_dataset = datasets.CIFAR10(
    #     root=data_dir, train=True,
    #     download=True, transform=valid_transform,
    # )


    train_dataset = ImageFolderWithPaths(root=data_dir,
                                         transform=train_transform
                                         )

    val_dataset = ImageFolderWithPaths(root=data_dir,
                                         transform=val_transform
                                         )


    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,  collate_fn=remove_none_collate
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,  collate_fn=remove_none_collate
    )

    return (train_loader, valid_loader)