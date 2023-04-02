import torch
import torch.nn as nn
import matplotlib.pyplot as pplt
import torch.nn.functional as F

class SoftMax:
    def __init__(self, x, y, c):
        self.x = torch.cat((torch.ones(x.shape[0], 1), x), dim=1)
        self.y = y
        self.c = c
        self.w = torch.zeros(c, x.shape[1]+1)

    def softmax(self, x_test=None):
        if x_test == None:
            x_test = self.x
        return torch.exp(x_test)/torch.exp(x_test).sum(dim=1).reshape(-1,1)

    def errors(self, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        return [i for i in range(len(x_test)) if torch.argmax(self.predict(x_test), dim=1)[i] != torch.argmax(y_test, dim=1)[i]]

    def predict(self, x_test=None):
        if x_test == None:
            x_test = self.x
        return self.softmax(x_test@self.w.t())

    def train(self, x_train=None, y_train=None):
        if x_train == None or y_train  ==None:
            x_train, y_train = self.x, self.y
        it = 0
        N = x_train.shape[0]
        accuracy_list = []
        while len(self.errors(x_train, y_train)) != 0:
            y_out = self.predict(x_train)
            #print(y_out)
            gd_w = ((y_out-y_train).t()@x_train).nan_to_num()
            self.w -= 0.01*gd_w
            #print((y_out-self.y).t(),self.x)
            #print(gd_w)
            #print(self.w)
            it += 1
            if it>1000:
                break
            accuracy = 1-(torch.argmax(self.predict(x_train), dim=1) != torch.argmax(y_train, dim=1)).int().sum()/N
            accuracy_list.append(accuracy)
        pplt.plot(range(len(accuracy_list)),accuracy_list,'-')
        pplt.show()
        print(self.w, accuracy)
        return self.w, accuracy

    def test(self, x_test, y_test):
        x_test = torch.cat((torch.ones(x_test.shape[0], 1), x_test), dim=1)
        accuracy = 1-(torch.argmax(self.predict(x_test), dim=1) != torch.argmax(y_test, dim=1)).int().sum()/len(x_test)
        print('test accuracy:', accuracy)
        return accuracy

class Softnn(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    def __init__(self, dim_in, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),#展开图像
            nn.Linear(dim_in, C),#全连接
            nn.ReLU(),#激活函数
            nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
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

def acc(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')   
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
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

            tt = e*len(loader_train)+t

        acc_history.append(acc(loader_train, model))
        acc_test_history.append(acc(loader_test, model))
        loss_history.append(loss.item())
        iter_history.append(tt)

    return acc_history, acc_test_history, loss_history, iter_history