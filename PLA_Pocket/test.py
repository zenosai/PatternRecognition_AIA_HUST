from PLA_Pocket import PLA_Pocket
import torch
import time

# test1
'''x1 = torch.randn(200, 2)+torch.tensor([-5, 0])
x2 = torch.randn(200, 2)+torch.tensor([0, -5])
train_num = int(200*0.8)
test_num = 200-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
x_train = torch.cat((x1_train, x2_train), dim=0)
y_train = torch.cat((torch.ones(train_num), -torch.ones(train_num)), dim=0)
x_test = torch.cat((x1_test, x2_test), dim=0)
y_test = torch.cat((torch.ones(test_num), -torch.ones(test_num)), dim=0)

time_start = time.process_time()
sample = PLA_Pocket(x_train, y_train)
w1, _ = sample.PLA_train()
time_end = time.process_time()
print("PLA run time:", time_end-time_start)
sample.test(x_test, y_test)

time_start = time.process_time()
sample = PLA_Pocket(x_train, y_train)
w2, _ = sample.Pocket_train()
time_end = time.process_time()
print("Pocket run time:", time_end-time_start)
sample.test(x_test, y_test)'''

# test2
x1 = torch.randn(1000, 2)+torch.tensor([1, 0])
x2 = torch.randn(1000, 2)+torch.tensor([0, 1])
train_num = int(1000*0.8)
test_num = 1000-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
x_train = torch.cat((x1_train, x2_train), dim=0)
y_train = torch.cat((torch.ones(train_num), -torch.ones(train_num)), dim=0)
x_test = torch.cat((x1_test, x2_test), dim=0)
y_test = torch.cat((torch.ones(test_num), -torch.ones(test_num)), dim=0)

time_start = time.process_time()
sample = PLA_Pocket(x_train, y_train)
w1, _ = sample.PLA_train()
time_end = time.process_time()
print("PLA run time:", time_end-time_start)
sample.test(x_test, y_test)

time_start = time.process_time()
sample = PLA_Pocket(x_train, y_train)
w2, _ = sample.Pocket_train()
time_end = time.process_time()
print("Pocket run time:", time_end-time_start)
sample.test(x_test, y_test)