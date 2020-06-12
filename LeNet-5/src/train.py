import torch
import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='../MNIST', train=True, download=True, transform=None)
