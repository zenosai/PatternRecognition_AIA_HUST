import torch
import matplotlib.pyplot as pplt
import random


class PLA_Pocket():
    def __init__(self, x, y, maxiter=1000, maxtimes_nochange=100):
        N, dim = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1)
        self.y = y
        self.w = torch.zeros(dim+1)
        self.maxiter = maxiter
        self.maxtimes_nochange = maxtimes_nochange

    def PLA_train(self):
        N, _ = self.x.shape
        err_index = self.errors(self.w)
        err = len(err_index)
        for i in range(self.maxiter):
            if not err:
                break
            self.w += self.y[err_index[0]]*self.x[err_index[0]]
            err_index = self.errors(self.w)
            err = len(err_index)
            print("times: ", i, ", accuracy: ",
                  float(1-err/N), ", w", self.w)
        rate = 1-err/N
        print("train: accuracy: ", float(rate))
        pplt.title("PLA trian")
        self.draw()
        return [self.w, rate]

    def Pocket_train(self):
        N, _ = self.x.shape
        err_index = self.errors(self.w)
        err = len(err_index)
        times_nochange = 0
        w_tmp = self.w
        for i in range(self.maxiter):
            if not err or times_nochange >= self.maxtimes_nochange:
                break
            err_index = self.errors(w_tmp)
            i_rand = err_index[random.randint(0, len(err_index)-1)]
            w_tmp = w_tmp + self.y[i_rand]*self.x[i_rand]
            err_tmp = len(self.errors(w_tmp))
            if err > err_tmp:
                self.w = w_tmp
                err = err_tmp
                times_nochange = 0
            else:
                times_nochange += 1
            print("times: ", i, ", accuracy: ",
                  float(1-err/N), ", w", self.w)
        accuracy = 1-err/N
        print("train: accuracy: ", float(accuracy))
        pplt.title("Pocket trian")
        self.draw()
        return [self.w, accuracy]

    def errors(self, w_test, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        N, _ = x_test.shape
        errors = []
        y_gx = torch.sign(torch.mv(x_test, w_test))
        for i in range(N):
            if (y_test[i] != y_gx[i]):
                errors.append(i)
        return errors

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
        rate = 1-err/N
        print("test: accuracy: ", float(rate))
        pplt.title("test")
        self.draw(x_ex, y_test)
