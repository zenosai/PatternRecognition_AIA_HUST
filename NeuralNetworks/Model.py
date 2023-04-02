import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as pplt
import math

class LinearBlock(nn.Module):
    def __init__(self, dim_in, dim_out, act='relu') -> None:
        super().__init__()
        if act=='relu':
            self.net = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU()
            )
        elif act=='sigmoid':
            self.net = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.Sigmoid()
            )
        elif act=='tanh':
            self.net = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.Tanh()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU()
            )

        
    def forward(self, x):
        return self.net(x)

class MulNet(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    # dim_h: 隐藏层神经元个数
    def __init__(self, dim_in, C, dim_h=8, num_blocks=2, block=LinearBlock, act='relu') -> None:
        super().__init__()
        blocks = [block(dim_in, dim_h, act)]
        for _ in range(num_blocks-2):
            blocks.append(block(dim_h, dim_h))
        blocks.append(block(dim_h, C))
        self.net = nn.Sequential(
            nn.Flatten(),
            *blocks,
            nn.Softmax()
        )
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.net(x)
        return score

class LeNet(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    # dim_h: 隐藏层神经元个数
    def __init__(self, C, H, W) -> None:
        super().__init__()
        assert(C==1)
        self.layer = nn.Sequential(
            #Conv-1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            #nn.Sigmoid(),
            #AvePool-1
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #Conv-2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            #nn.Sigmoid(),
            #AvePool-2
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #FC-1
            nn.Flatten(),
            nn.Linear(int(16*(H/4-2)*(W/4-2)), 120),
            nn.Sigmoid(),
            #FC-2
            nn.Linear(120, 84),
            nn.Sigmoid(),
            #out
            nn.Linear(84, 10),
            nn.Sigmoid(),
            nn.Softmax())
        # 初始化策略
        for node in self.layer:
            if isinstance(node, nn.Linear):
                nn.init.xavier_normal_(node.weight)
                # nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(node.bias, 0)
            elif isinstance(node, nn.Conv2d):
                nn.init.kaiming_normal_(node.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(node, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(node.weight, 1)
                nn.init.constant_(node.bias, 0)

    def forward(self, x):
        x = self.layer(x)
        return x

def show_result(acc_history, acc_test_history, loss_history, iter_history):
    fig = pplt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(iter_history, acc_history, 'r-', label='train acc')
    ax.plot(iter_history, acc_test_history, 'b-', label='test acc')
    ax2.plot(iter_history, loss_history, 'c-', label='loss')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.set_ylabel(r"acc")
    ax2.set_ylabel(r"loss")
    pplt.show()

def show_mnist_sample(model, loader, batch_size=64, num=10, row=5):
    if loader.dataset.train:
        print('show random sample on train set')
    else:
        print('show random sample on test set')
    full_num = batch_size
    sample = list(torch.utils.data.WeightedRandomSampler(torch.ones(full_num)/full_num,num_samples=num, replacement=False))
    examples = enumerate(loader)
    _, (example_data, example_targets) = next(examples) # 实际上只是在第一个batch里抽样
    fig = pplt.figure(figsize=(8, 6))
    for i in range(num):
        pplt.subplot(math.ceil(num/row), row, i+1)
        # plt.tight_layout()
        pplt.imshow(example_data[sample[i]][0], cmap='gray', interpolation='none')
        pplt.title("Pre: {}, act: {}".format((model(example_data)[sample[i]]).argmax(), example_targets[sample[i]]))
        pplt.xticks([])
        pplt.yticks([])
    pplt.show()

def accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')   
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # to avoid error, ensure that x(input) is float, y(target) is long
            # and call loss on model output and y without any typecasting
            x=x.float()
            y=y.long()
            scores = model(x)
            _, pred_y = scores.max(1)
            correct += (pred_y==y).sum()
            num += pred_y.size(0)
        rate = correct.float()/num
        print('Got %d / %d correct (%.2f)' % (correct, num, 100 * rate))
    return rate

def train(model, loader_train, loader_test, optimizer, epoch=1):
    acc_history = []
    acc_test_history = []
    loss_history = []
    iter_history = []
    for e in range(epoch):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.float()
            y = y.long()
            score = model(x)
            loss = F.cross_entropy(score, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_history.append(accuracy(loader_train, model))
        acc_test_history.append(accuracy(loader_test, model))
        loss_history.append(loss.item())
        iter_history.append(e)

    return acc_history, acc_test_history, loss_history, iter_history