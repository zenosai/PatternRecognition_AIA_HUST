import torch
import random

class PLA_Pocket():
    def __init__(self, x, y, maxiter=1000, maxtimes_nochange=100):
        N, dim = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1)
        self.y = y
        self.w = torch.zeros(dim+1)
        self.maxiter = maxiter
        self.maxtimes_nochange = maxtimes_nochange

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