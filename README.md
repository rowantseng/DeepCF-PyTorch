# DeepCF-PyTorch

This is a PyTorch implementation of the paper:

Zhi-Hong Deng, Ling Huang, Chang-Dong Wang, Jian-Huang Lai, Philip S. Yu. [DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System](https://arxiv.org/abs/1901.04704v1). In AAAI '19(Oral), Honolulu, Hawaii, USA, January 27 – February 1, 2019.

`Dataset.py` and `evaluate.py` is borrowed from the original implementation [here](https://github.com/familyld/DeepCF). It is worth it to mention that line 63-80 in `evaluate.py` is revised for PyTorch inference.

## Installation

```shell
pip install --no-cache-dir -r requirements.txt
```

## Dataset

The authors provide all four processed datasets: MovieLens 1 Million (ml-1m), LastFm (lastfm), Amazon Toy (AToy) and Amazon Music (AMusic).

train.rating
- Train file.
- Each Line is a training instance: `userID\t itemID\t rating\t timestamp (if have)`

test.rating
- Test file (positive instances).
- Each Line is a testing instance: `userID\t itemID\t rating\t timestamp (if have)`

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.
- Each line is in the format: `(userID,itemID)\t negativeItemID1\t negativeItemID2 ...`

## Train and Evaluate

```shell
# Train CFNet-rl
python DMF.py --dataset ml-1m

# Train CFNet-ml
python MLP.py --dataset ml-1m

# Train CFNet w/o pretrained weights
python CFNet.py --dataset ml-1m --lr 0.01

# Train CFNet w/ pretrained weights
python CFNet.py --dataset ml-1m --dmf /path/to/dmf/model --mlp /path/to/mlp/model --lr 0.00001
```

The evaluation results are shown as below:

|                              | Hit Rate(HR) | NDCG         |
|------------------------------|--------------|--------------|
| CFNet-rl (DMF)               | 0.7055       | 0.4212       |
| CFNet-ml (MLP)               | 0.7055       | 0.4236       |
| CFNet w/o Pretrained Weights | 0.7048       | 0.4249       |
| CFNet w/ Pretrained Weights  | 0.7220       | 0.4388       |

There is one thing to be noticed that the CFNet with pretrained weights is saved at initial point. We don't actually need to fintune the model and the hit rate is equal to 0.7220 which is very close to 0.7253 reported on the paper(while NDCG is 0.4416 on the paper). By copying the weights from `CFNet-rl(DMF)` and `CFNet-ml(MLP)` models, CFNet has better representation on capturing stronger expressiveness rather than single model trained on tasks, representation learning(`CFNet-rl`) or matching function(`CFNet-ml`). So far, I have trained CFNet using several hyperparameters, but the model saved at initialization has the best performance while the loss decreased over epochs. Below is the screenshot of the experimental results:

```shell
DeepCF arguments: Namespace(bsz=256, dataset='ml-1m', dmf='pretrained/ml-1m_DMF_1627953521.6537452.pth', epochs=20, fcLayers='[512, 256, 128, 64]', itemLayers='[1024, 64]', lr=1e-05, mlp='pretrained/ml-1m_MLP_1627958360.4192731.pth', nNeg=4, optim='adam', path='Data/', userLayers='[512, 64]') 
Use CUDA? True
Load data: #user=6040, #item=3706, #train=994169, #test=6040 [16.5s]
Load pretrained DMF from pretrained/ml-1m_DMF_1627953521.6537452.pth
Load pretrained MLP from pretrained/ml-1m_MLP_1627958360.4192731.pth
Init: HR=0.7220, NDCG=0.4388 [28.2s]
Epoch 1: Loss=0.1838 [407.0s]
Epoch 1: HR=0.7098, NDCG=0.4369 [28.3s]
Epoch 2: Loss=0.1800 [407.2s]
Epoch 2: HR=0.7040, NDCG=0.4350 [28.4s]
Epoch 3: Loss=0.1779 [405.9s]
Epoch 3: HR=0.7020, NDCG=0.4330 [28.3s]
Epoch 4: Loss=0.1765 [405.8s]
Epoch 4: HR=0.7023, NDCG=0.4332 [28.0s]
Epoch 5: Loss=0.1753 [406.6s]
Epoch 5: HR=0.7028, NDCG=0.4332 [28.2s]
...
```

## Citation

Please cite authors of paper if you use the codes. Thanks!

```
@article{deng2019deepcf,
  title={DeepCF: A Unified Framework of Representation Learning and Matching Function Learning in Recommender System},
  author={Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Yu, Philip S},
  booktitle={AAAI},
  year={2019}
}

@misc{deepcf2019,
  author =       {Deng, Zhi-Hong and Huang, Ling and Wang, Chang-Dong and Lai, Jian-Huang and Yu, Philip S},
  title =        {DeepCF},
  howpublished = {\url{https://github.com/familyld/DeepCF}},
  year =         {2019}
}
```