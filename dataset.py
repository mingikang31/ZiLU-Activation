'''CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 




# ### WIKI Text 103 
# train_path = ('./Data/train.bin')
# train_data = np.memmap(train_path, dtype='uint16', mode='r')

# batch_size = 12 
# block_size = 1028 
# bias = False 
# real_data = True

# compile = True 

# def get_batch(split):
#     data = train_data 
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix])

#     x, y = x.to('cuda'), y.to('cuda')
#     return x, y


# # Training script: 
    
    
# // ...existing code...
# from torch.utils.data import Dataset, DataLoader

# class WikiText103Dataset(Dataset):
#     def __init__(self, data_path, block_size):
#         self.data = np.memmap(data_path, dtype='uint16', mode='r')
#         self.block_size = block_size
        
#     def __len__(self):
#         return len(self.data) - self.block_size
    
#     def __getitem__(self, idx):
#         x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
#         y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
#         return x, y

# class WikiText103:
#     def __init__(self, args=None, batch_size=12, block_size=1024, data_dir='./Data'):
#         """
#         Simple WikiText-103 dataset wrapper
        
#         Args:
#             args: Optional args object (for compatibility with CIFAR classes)
#             batch_size: Batch size for data loaders
#             block_size: Sequence length
#             data_dir: Directory containing train.bin and val.bin
#         """
#         # Handle args if provided (for consistency with CIFAR classes)
#         if args is not None:
#             batch_size = getattr(args, 'batch_size', batch_size)
#             block_size = getattr(args, 'block_size', block_size)
#             data_dir = getattr(args, 'data_path', data_dir)
        
#         self.batch_size = batch_size
#         self.block_size = block_size
#         self.vocab_size = 50257  # GPT-2 vocab size
        
#         # Create datasets
#         train_path = f'{data_dir}/train.bin'
#         val_path = f'{data_dir}/val.bin'
        
#         self.train_data = WikiText103Dataset(train_path, block_size)
#         self.test_data = WikiText103Dataset(val_path, block_size)  # Using val as test for consistency
        
#         # Create data loaders
#         self.train_loader = DataLoader(
#             self.train_data, 
#             batch_size=batch_size, 
#             shuffle=True, 
#             num_workers=4,
#             pin_memory=True
#         )
        
#         self.test_loader = DataLoader(
#             self.test_data, 
#             batch_size=batch_size, 
#             shuffle=False, 
#             num_workers=4,
#             pin_memory=True
#         )
        
#         # For compatibility
#         self.num_classes = self.vocab_size
    
#     def shape(self):
#         """Return input shape for compatibility"""
#         return (self.block_size,)  # Sequence length
    
#     def get_batch(self, split='train'):
#         """Get a single batch - alternative to using DataLoader"""
#         if split == 'train':
#             data = self.train_data.data
#         else:
#             data = self.test_data.data
            
#         ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
#         x = torch.stack([torch.from_numpy((data[i: i + self.block_size]).astype(np.int64)) for i in ix])
#         y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + self.block_size]).astype(np.int64)) for i in ix])
#         return x, y

# // ...existing code...


'''Wikitext-103 Dataset Class'''
class WikiText103:
    def __init__(self, args):
        pass 

    
    

'''CIFAR-10 and CIFAR-100 Dataset Classes'''
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        self.CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        self.CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [
                transforms.Resize((args.resize, args.resize))
                ]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
            
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 100

    def shape(self):
        return self.train_data[0][0].shape
        
class CIFAR10(datasets.CIFAR10): 
    def __init__(self, args):

        self.CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR10_STD = (0.2470, 0.2435, 0.2616)

        # Transformations
        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 10

    def shape(self): 
        return self.train_data[0][0].shape
    