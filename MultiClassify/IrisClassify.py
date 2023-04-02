from re import I
import torch
import matplotlib.pyplot as pplt
from sklearn.datasets import load_iris
import random
from OVO import OVO
from Softmax import SoftMax

############ test ################
if __name__ == '__main__':
    data = load_iris()
    x = torch.tensor(data.data)
    y = torch.tensor(data.target)

    x = x.numpy().tolist()
    y = y.numpy().tolist()
    tmp = list(zip(x, y))
    random.shuffle(tmp)
    x, y = zip(*tmp)
    x = torch.tensor(x)
    y = torch.tensor(y)

    train_num = 30
    test_num = 20
    train_x = x[:train_num]
    train_y = y[:train_num]
    test_x = x[train_num:train_num+test_num]
    test_y = y[train_num:train_num+test_num]

    # ovo
    model = OVO(train_x, train_y, 3)
    model.train()
    model.test(test_x, test_y)

    #softmax
    y_soft = torch.zeros(len(train_y), 3)
    for i in range(len(train_y)):
        y_soft[i, train_y[i]] = 1
    model = SoftMax(train_x, y_soft, 3)
    model.train()
    y_test_soft = torch.zeros(len(test_y), 3)
    for i in range(len(test_y)):
        y_test_soft[i, test_y[i]] = 1
    model.test(test_x, y_test_soft)
