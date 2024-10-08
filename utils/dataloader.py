import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.sampler import BatchSampler
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, random_split, Subset
from collections.abc import Mapping, Sequence
from medmnist import PathMNIST, ChestMNIST, OCTMNIST # 导入 PathMNIST 数据集类

from utils.data_aug import CIFARPolicy, SVHNPolicy, ImageNetPolicy, RandomErasing

class RASampler(torch.utils.data.Sampler):
    """
    Batch Sampler with Repeated Augmentations (RA)
    - dataset_len: original length of the dataset
    - batch_size
    - repetitions: instances per image
    - len_factor: multiplicative factor for epoch size
    """

    def __init__(self, dataset_len, batch_size, repetitions=1, len_factor=3.0, shuffle=False, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def shuffler(self):
        if self.shuffle:
            new_perm = lambda: iter(np.random.permutation(self.dataset_len))
        else:
            new_perm = lambda: iter(np.arange(self.dataset_len))
        shuffle = new_perm()
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            for repetition in range(self.repetitions):
                yield index

    def __iter__(self):
        shuffle = iter(self.shuffler())
        seen = 0
        batch = []
        for _ in range(self.len_images):
            index = next(shuffle)
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.len_images // self.batch_size
        else:
            return (self.len_images + self.batch_size - 1) // self.batch_size

def list_collate(batch):
    """
    Collate into a list instead of a tensor to deal with variable-sized inputs
    """
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy':
        if elem_type.__name__ == 'ndarray':
            return list_collate([torch.from_numpy(b) for b in batch])
    elif isinstance(batch[0], Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    return default_collate(batch)      

class CifarLoader:
    def __init__(self, config):
        self.data_root = os.path.join(config['data_root'], config['cifar_type'])
        self.cifar_type = config['cifar_type']
        self.valid_scale = config['valid_scale']
        
        self.batch_size = config['batch_size']
        self.img_size = config['img_size']
        self.norm = config['norm']
        
        self.mp = config['multi_process']
        
        self.num_classes = config['num_classes']
        
        # augmentation
        if 'augmentation' in config:
            aug = []
            aug_config = config['augmentation']
            aug += [transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.img_size, padding=4)]
            
            if 'aug_policy' in aug_config:
                if aug_config['aug_policy'] == 'CIFAR':
                    aug_policy = CIFARPolicy()
                elif aug_config['aug_policy'] == 'SVHN':
                    aug_policy = SVHNPolicy()
                elif aug_config['aug_policy'] == 'ImageNet':
                    aug_policy = ImageNetPolicy()
                aug += [aug_policy]
            
            aug += [transforms.ToTensor(), 
                    transforms.Normalize(self.norm[0], self.norm[1])]
            
            if 'random_erasing' in aug_config:
                re_config = aug_config['random_erasing']
                re = RandomErasing(re_config['prob'], sh=re_config['sh'], r1=re_config['r1'], mean=self.norm[0])
                aug += [re]
            
            train_transform = transforms.Compose(aug)
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1]),
            ])   
        
        
        val_test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1]),
        ]) 
        
        if self.cifar_type == 'CIFAR10':      
            trainset = torchvision.datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=train_transform)
            testset  = torchvision.datasets.CIFAR10(root=self.data_root, train=False, download=True, transform=val_test_transform)
            self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck')
                            
        elif self.cifar_type == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(root=self.data_root, train=True, download=True, transform=train_transform)
            testset  = torchvision.datasets.CIFAR100(root=self.data_root, train=False, download=True, transform=val_test_transform)
            self.classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 
                            'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
                            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 
                            'camel', 'can', 'castle', 'caterpillar', 'cattle', 
                            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
                            'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 
                            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 
                            'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 
                            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 
                            'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 
                            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 
                            'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 
                            'plain', 'plate', 'poppy', 'porcupine', 'possum', 
                            'rabbit', 'raccoon', 'ray', 'road', 'rocket', 
                            'rose', 'sea', 'seal', 'shark', 'shrew', 
                            'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
                            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
                            'tank', 'telephone', 'television', 'tiger', 'tractor', 
                            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 
                            'whale', 'willow_tree', 'wolf', 'woman', 'worm')

        if self.valid_scale != 0:
            train_size, val_size = len(trainset) * (1 - self.valid_scale), len(trainset) * self.valid_scale
            train_, valid_ = torch.utils.data.random_split(trainset, [int(train_size), int(val_size)])

            self.trainloader = torch.utils.data.DataLoader(train_,  num_workers=self.mp, pin_memory=True,
                batch_sampler=RASampler(len(train_), self.batch_size, 1, aug_config['repeat_aug'], shuffle=True, drop_last=True))
            self.validloader = torch.utils.data.DataLoader(valid_, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
            self.testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
        else:
            self.trainloader = torch.utils.data.DataLoader(trainset,  num_workers=self.mp, pin_memory=True,
                batch_sampler=RASampler(len(trainset), self.batch_size, 1, aug_config['repeat_aug'], shuffle=True, drop_last=True))
            self.validloader = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
            self.testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)

class ImagenetLoader:
    def __init__(self, config):
        self.data_root = os.path.join(config['data_root'])
        self.imagenet_type = config['imagenet_type']
        self.valid_scale = config['valid_scale']
        
        self.batch_size = config['batch_size']
        self.img_size = config['img_size']
        self.norm = config['norm']
        
        self.mp = config['multi_process']
        
        self.num_classes = config['num_classes']
        
        # augmentation
        if 'augmentation' in config:
            aug = []
            aug_config = config['augmentation']
            aug += [transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.img_size, padding=4)]
            
            if 'aug_policy' in aug_config:
                if aug_config['aug_policy'] == 'CIFAR':
                    aug_policy = CIFARPolicy()
                elif aug_config['aug_policy'] == 'SVHN':
                    aug_policy = SVHNPolicy()
                elif aug_config['aug_policy'] == 'ImageNet':
                    aug_policy = ImageNetPolicy()
                aug += [aug_policy]
            
            aug += [transforms.ToTensor(), 
                    transforms.Normalize(self.norm[0], self.norm[1])]
            
            if 'random_erasing' in aug_config:
                re_config = aug_config['random_erasing']
                re = RandomErasing(re_config['prob'], sh=re_config['sh'], r1=re_config['r1'], mean=self.norm[0])
                aug += [re]
            
            train_transform = transforms.Compose(aug)
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(self.img_size, padding=4),
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.norm[0], self.norm[1]),
            ])   
        
        
        val_test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.norm[0], self.norm[1]),
        ]) 
        
        if self.imagenet_type == 'Tiny':      
            trainset = torchvision.datasets.ImageFolder(root=os.path.join(self.data_root, 'train'), transform=train_transform)
            testset  = torchvision.datasets.ImageFolder(root=os.path.join(self.data_root, 'val'), transform=val_test_transform)
            
                            
        elif self.imagenet_type == 'Complete':
            pass

        if self.valid_scale != 0:
            train_size, val_size = len(trainset) * (1 - self.valid_scale), len(trainset) * self.valid_scale
            train_, valid_ = torch.utils.data.random_split(trainset, [int(train_size), int(val_size)])

            self.trainloader = torch.utils.data.DataLoader(train_,  num_workers=self.mp, pin_memory=True,
                batch_sampler=RASampler(len(train_), self.batch_size, 1, aug_config['repeat_aug'], shuffle=True, drop_last=True))
            self.validloader = torch.utils.data.DataLoader(valid_, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
            self.testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
        else:
            self.trainloader = torch.utils.data.DataLoader(trainset,  num_workers=self.mp, pin_memory=True,
                batch_sampler=RASampler(len(trainset), self.batch_size, 1, aug_config['repeat_aug'], shuffle=True, drop_last=True))
            self.validloader = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)
            self.testloader  = torch.utils.data.DataLoader(testset,  batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.mp)

