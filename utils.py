import numpy as np

import torch


class AverageMeter(object):
    """Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=".4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr


class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput, labels):
        super(BatchDataset, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput):
        super(TestDataset, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index]

    def __len__(self):
        return self.userInput.size(0)


def get_optimizer(name, lr, scope):
    if name.lower() == "adagrad":
        return torch.optim.Adagrad(scope, lr=lr)
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop(scope, lr=lr)
    elif name.lower() == "adam":
        return torch.optim.Adam(scope, lr=lr)
    elif name.lower() == "sgd":
        return torch.optim.SGD(scope, lr=lr)
    else:
        raise ValueError(f"{name} optimizer is not supported!")


def get_train_matrix(train):
    nUsers, nItems = train.shape
    trainMatrix = np.zeros([nUsers, nItems], dtype=np.int32)
    for (u, i) in train.keys():
        trainMatrix[u][i] = 1
    return trainMatrix


def get_train_instances(train, nNeg):
    userInput, itemInput, labels = [], [], []
    nUsers, nItems = train.shape
    for (u, i) in train.keys():
        # positive instance
        userInput.append(u)
        itemInput.append(i)
        labels.append(1)
        # negative instances
        for t in range(nNeg):
            j = np.random.randint(nItems)
            while (u, j) in train.keys():
                j = np.random.randint(nItems)
            userInput.append(u)
            itemInput.append(j)
            labels.append(0)
    return userInput, itemInput, labels
