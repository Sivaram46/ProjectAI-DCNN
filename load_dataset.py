import torch 
from torch.utils.data import DataLoader 

from torchvision import datasets
from torchvision.transforms import ToTensor

def load_fmnist(batch_size: int):
    """
    Download and returns train and test dataloader for FashionMNIST dataset.
    Datapoints in train set: 60,000
    Datapoints in test set : 10,000
    No of classes: 10 
    """
    return _load_data(batch_size, 'FashionMNIST')

def load_mnist(batch_size: int):
    """
    Download and returns train and test dataloader for MNIST dataset.
    Datapoints in train set: 60,000
    Datapoints in test set : 10,000
    No of classes: 10 
    """
    return _load_data(batch_size, 'MNIST')

def load_cifar10(batch_size: int):
    """
    Download and returns train and test dataloader for CIFAR10 dataset.
    Datapoints in train set: 50,000
    Datapoints in test set : 10,000
    No of classes: 10
    """
    return _load_data(batch_size, 'CIFAR10'), classes

def load_cifar100(batch_size: int):
    """
    Download and returns train and test dataloader for CIFAR100 dataset.
    Datapoints in train set: 50,000
    Datapoints in test set : 10,000
    No of classes: 100
    """
    return _load_data(batch_size, 'CIFAR100')

def _load_data(batch_size, name):
    if(name == 'FashionMNIST'):
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=ToTensor()
        )

        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=ToTensor()
        )

        classes = (
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
            'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        )

    elif(name == 'MNIST'):
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=ToTensor()
        )

        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=ToTensor()
        )

        classes = tuple([str(i) for i in range(10)])
    
    elif(name == 'CIFAR10'):
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=ToTensor()
        )

        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=ToTensor()
        )

        classes = (
            'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 
            'truck'
        )
    
    elif(name == 'CIFAR100'):
        train_data = datasets.CIFAR100(
            root='data', train=True, download=True, transform=ToTensor()
        )

        test_data = datasets.CIFAR100(
            root='data', train=False, download=True, transform=ToTensor()
        )

        classes = None
    
    else:
        raise ValueError("No such dataset")

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    return train_dataloader, test_dataloader, classes