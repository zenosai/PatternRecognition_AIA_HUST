import torch
from FisherLinearDiscriminant import FLD

# test1
x1 = torch.randn(200, 2)+torch.tensor([-5, 0])
x2 = torch.randn(200, 2)+torch.tensor([0, -5])
train_num = int(200*0.8)
test_num = 200-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
sample = FLD(x1_train, x2_train)
w, s = sample.FLDA()
sample.test(x1_test,x2_test)