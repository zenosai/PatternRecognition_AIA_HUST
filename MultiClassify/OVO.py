import torch
from LogitsitcRegression import LogitsticRegression
from PLA_Pocket import PLA_Pocket

class OVO:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.c = c
        self.w = []

    def train(self):
        for i in range(self.c):
            for j in range(i):  # 下三角
                x1 = self.x[self.y==i]
                x2 = self.x[self.y==j]
                x = torch.cat((x1, x2), dim=0)
                y = torch.cat((torch.ones(len(x1)), -torch.ones(len(x2))))
                #model = LogitsticRegression(x, y) #Logiststic回归效果不好
                #w = model.gd_train(a = 0.01, epoch_num=100)[0]
                model = PLA_Pocket(x, y, 200, 100)
                w = model.Pocket_train()[0]
                print(w)
                self.w.append(w.numpy())
        self.w = torch.tensor(self.w).t()
        x = torch.cat((torch.ones(len(self.x),1), self.x), dim=1)
        vote = torch.zeros(len(x), self.c)
        for i in range(1,self.c):
            for j in range(i):
                tmp = x@self.w
                vote[:, i] += (tmp[:, int(i*(i-1)/2+j)])>0
                vote[:, j] += (tmp[:, int(i*(i-1)/2+j)])<0
        y_out = torch.argmax(vote, dim=1).int()
        accuracy = sum(y_out==self.y)/len(x)
        print(self.w)
        print("OVO train accuracy: ",accuracy)
        return self.w

    def test(self, x_test, y_test):
        vote = torch.zeros(len(x_test), self.c)
        x_test = torch.cat((torch.ones(x_test.shape[0],1), x_test), dim=1)
        for i in range(1,self.c):
            for j in range(i):
                tmp = x_test@self.w
                vote[:, i] += (tmp[:, int(i*(i-1)/2+j)])>0
                vote[:, j] += (tmp[:, int(i*(i-1)/2+j)])<0
        y_out = torch.argmax(vote, dim=1).int()
        accuracy_test = sum(y_out==y_test).int()/len(x_test)
        print("OVO test accuracy: ",accuracy_test)