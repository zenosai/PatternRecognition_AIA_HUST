import torch
import matplotlib.pyplot as pplt


class LinearRegression():
    def __init__(self, x, y):
        N, dim = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1)
        self.y = y
        self.w = torch.zeros(dim+1)

    def errors(self, w_test, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        N, _ = x_test.shape
        y_gx = torch.sign(torch.mv(x_test, w_test))
        return [i for i in range(N) if y_test[i] != y_gx[i]]

    def generalized_inverse_method(self):
        self.w = torch.mv(torch.mm(torch.linalg.inv(
            torch.mm(self.x.t(), self.x)), self.x.t()), self.y)
        accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
        self.draw()
        print("train: accuracy: ", float(accuracy))
        return self.w, accuracy

    def gd_train(self, a=0.05, epoch_num=1000, batch=100):
        N = len(self.x)
        loss = []
        for i in range(epoch_num):
            for j in range((N-1)//batch+1):
                x_batch = self.x[batch*j:min(batch*(j+1), N), :]
                y_batch = self.y[batch*j:min(batch*(j+1), N)]
                #print(x_batch, y_batch)
                gd_w = 2*torch.mv(x_batch.t(),
                                  torch.mv(x_batch, self.w) - y_batch)/len(x_batch)
                self.w -= a * gd_w
                #accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
            loss.append(self.loss())
        pplt.figure(1)
        pplt.plot(loss)
        pplt.figure(2)
        self.draw()
        accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
        print("train: accuracy: ", float(accuracy))
        return self.w, accuracy

    def loss(self):
        return torch.norm(torch.mv(self.x, self.w)-self.y)**2

    def draw(self, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        y = y_test
        err_indexes = self.errors(self.w, x_test, y_test)

        x_pos_right = x_test[[index for index in range(
            len(x_test)) if index not in err_indexes and y[index] == 1]]
        x_neg_right = x_test[[index for index in range(
            len(x_test)) if index not in err_indexes and y[index] == -1]]

        x_pos_false = x_test[[index for index in err_indexes if y[index] == 1]]
        x_neg_false = x_test[[
            index for index in err_indexes if y[index] == -1]]

        pplt.plot(x_pos_right[:, 1], x_pos_right[:, 2], 'bo')
        pplt.plot(x_neg_right[:, 1], x_neg_right[:, 2], 'bx')

        pplt.plot(x_pos_false[:, 1], x_pos_false[:, 2], 'ro')
        pplt.plot(x_neg_false[:, 1], x_neg_false[:, 2], 'rx')

        x_line = torch.tensor([min(x_test[:, 1]), max(x_test[:, 1])])
        y_line = -1*x_line*self.w[1]/self.w[2]-self.w[0]/self.w[2]
        pplt.plot(x_line, y_line, '-')
        pplt.show()

    def test(self, x_test, y_test):
        N,_ = x_test.shape
        x_ex = torch.cat((torch.ones(N).reshape(N, 1), x_test), dim=1)
        err_index = self.errors(self.w, x_ex, y_test)
        err = len(err_index)
        accuracy = 1-err/N
        print("test: accuracy: ", float(accuracy))
        pplt.title("test")
        self.draw(x_ex, y_test)


'''D = torch.tensor([[0.2, 0.7, 1],
                  [0.3, 0.3, 1],
                  [0.4, 0.5, 1],
                  [0.6, 0.5, 1],
                  [0.1, 0.4, 1],
                  [0.4, 0.6, -1],
                  [0.6, 0.2, -1],
                  [0.7, 0.4, -1],
                  [0.8, 0.6, -1],
                  [0.7, 0.5, -1]])
x = D[:, :2]
y = D[:, 2]

model = LinearRegression(x, y)
print(model.generalized_inverse_method())
print(model.gd_train(0.01, 100, 5))'''
