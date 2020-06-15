import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as tfs
from model.LeNet5 import LeNet5
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from matplotlib.pyplot import subplot


timg = tfs.Compose(
    [
        tfs.ToTensor(),
    ]
)

mnist_trainset = datasets.MNIST(root='./', train=True, download=True, transform=timg)
mnist_testset = datasets.MNIST(root='./', train=False, download=True, transform=timg)

idx = list(range(len(mnist_trainset)))

np.random.seed(1009)
np.random.shuffle(idx)

train_idx = idx[ : int(0.8 * len(idx))]
valid_idx = idx[int(0.8 * len(idx)) : ]

train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=30, sampler=train_set, num_workers=4)
valid_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=30, sampler=valid_set, num_workers=4)
test_loader = torch.utils.data.DataLoader(mnist_testset, num_workers=4)


net = LeNet5()
net.cuda()

loss_func = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

numEpochs = 20
training_accuracy = []
validation_accuracy = []

for epoch in range(numEpochs):
    epoch_training_loss = 0.0
    num_batches = 0

    for batch_num, training_batch in enumerate(train_loader):
        inputs, labels = training_batch
        inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())
        optim.zero_grad()
        forward_output = net(inputs)
        loss = loss_func(forward_output, labels)
        loss.backward()
        optim.step()
        epoch_training_loss += loss.data[0]
        num_batches += 1

    print("epoch: ", epoch, ", loss: ", epoch_training_loss/num_batches)

    accuracy = 0.0
    num_batches = 0

    for batch_num, training_batch in enumerate(train_loader):

        num_batches += 1
        inputs, actual_val = training_batch
        predicted_val = net(torch.autograd.Variable(inputs.cuda()))
        predicted_val = predicted_val.cpu().data.numpy()
        predicted_val = np.argmax(predicted_val, axis = 1)
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)

    validation_accuracy.append(accuracy/num_batches)


epochs = list(range(numEpochs))
fig1 = pyplot.figure()
pyplot.plot(epochs, training_accuracy, 'r')
pyplot.plot(epochs, validation_accuracy, 'g')
pyplot.xlabel("Epochs")
pyplot.ylabel("Accuracy")
pyplot.show(fig1)


correct = 0
total = 0
for test_data in test_loader:
    total += 1
    inputs, actual_val = test_data
    # perform classification
    predicted_val = net(torch.autograd.Variable(inputs.cuda()))
    # convert 'predicted_val' GPU tensor to CPU tensor and extract the column with max_score
    predicted_val = predicted_val.cpu().data
    max_score, idx = torch.max(predicted_val, 1)
    # compare it with actual value and estimate accuracy
    correct += (idx == actual_val).sum()

print("Classifier Accuracy: ", correct/total * 100)
