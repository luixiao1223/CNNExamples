import torch
import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='./', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./', train=False, download=True, transform=None)

for item in mnist_trainset:
    img, num = item[0], item[1]
