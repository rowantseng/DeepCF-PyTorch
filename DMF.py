import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
from Dataset import Dataset
from evaluate import evaluate_model
from utils import (AverageMeter, BatchDataset, get_optimizer,
                   get_train_instances, get_train_matrix)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF.")
    parser.add_argument("--path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",
                        help="Choose a dataset.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs.")
    parser.add_argument("--bsz", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--userLayers", nargs="?", default="[512, 64]",
                        help="Size of each user layer")
    parser.add_argument("--itemLayers", nargs="?", default="[1024, 64]",
                        help="Size of each item layer")
    parser.add_argument("--nNeg", type=int, default=4,
                        help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate.")
    parser.add_argument("--optim", nargs="?", default="adam",
                        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    return parser.parse_args()


class DMF(nn.Module):
    def __init__(self, userLayers, itemLayers, userMatrix, itemMatrix):
        super(DMF, self).__init__()
        assert userLayers[-1] == itemLayers[-1], "The last layer size of 'userLayers' and 'itemLayers' must be the same!"
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)

        # In the official implementation, 
        # the first dense layer has no activation
        layers = []
        layers.append(nn.Linear(nItems, userLayers[0]))
        for l1, l2 in zip(userLayers[:-1], userLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=True))
        self.userModel = nn.Sequential(*layers)
        layers = []
        layers.append(nn.Linear(nUsers, itemLayers[0]))
        for l1, l2 in zip(itemLayers[:-1], itemLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=True))
        self.itemModel = nn.Sequential(*layers)

        # In the official implementation, 
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(userLayers[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item):
        userInput = self.userMatrix[user, :]         # (B, 3706)
        itemInput = self.itemMatrix[item, :]         # (B, 6040)
        userVector = self.userModel(userInput)       # (B, userLayers[-1])
        itemVector = self.itemModel(itemInput)       # (B, itemLayers[-1])
        y = userVector * itemVector                  # element-wise multiplication
        y = self.final(y)                            # (B, 1)
        return y


if __name__ == "__main__":
    args = parse_args()
    userLayers = eval(args.userLayers)
    itemLayers = eval(args.itemLayers)
    topK = 10
    
    print("DMF arguments: %s " %(args))
    os.makedirs("pretrained", exist_ok=True)
    modelPath = f"pretrained/{args.dataset}_DMF_{time.time()}.pth"

    isCuda = torch.cuda.is_available()
    print(f"Use CUDA? {isCuda}")

    # Loading data
    t1 = time.time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    nUsers, nItems = train.shape
    print(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    
    # Build model
    userMatrix = torch.Tensor(get_train_matrix(train))
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    if isCuda:
        userMatrix, itemMatrix = userMatrix.cuda(), itemMatrix.cuda()
    
    model = DMF(userLayers, itemLayers, userMatrix, itemMatrix)
    if isCuda:
        model.cuda()
    torch.save(model.state_dict(), modelPath)
    
    optimizer = get_optimizer(args.optim, args.lr, model.parameters())
    criterion = torch.nn.BCELoss()

    # Check Init performance
    t1 = time.time()
    hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, num_thread=1)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print(f"Init: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
    bestHr, bestNdcg, bestEpoch = hr, ndcg, -1
    
    # Train model
    model.train()
    for epoch in range(args.epochs):
        t1 = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)
        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.bsz, shuffle=True)
        losses = AverageMeter("Loss")
        for ui, ii, lbl in ldr:
            if isCuda:
                ui, ii, lbl = ui.cuda(), ii.cuda(), lbl.cuda()
            ri = model(ui, ii).squeeze()
            loss = criterion(ri, lbl)

            # Update model and loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), lbl.size(0))

        print(f"Epoch {epoch+1}: Loss={losses.avg:.4f} [{time.time()-t1:.1f}s]")

        # Evaluation
        t1 = time.time()
        hits, ndcgs = evaluate_model(model, testRatings, testNegatives, topK, num_thread=1)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print(f"Epoch {epoch+1}: HR={hr:.4f}, NDCG={ndcg:.4f} [{time.time()-t1:.1f}s]")
        if hr > bestHr:
            bestHr, bestNdcg, bestEpoch = hr, ndcg, epoch
            torch.save(model.state_dict(), modelPath)

    print(f"Best epoch {bestEpoch+1}:  HR={bestHr:.4f}, NDCG={bestNdcg:.4f}")
    print(f"The best DMF model is saved to {modelPath}")
