import torch
import matplotlib.pyplot as pplt


class LogitsticRegression():
    def __init__(self, x, y):
        N, dim = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1)
        self.y = y
        self.w = torch.zeros(dim+1)

    def errors(self, w_test, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        N, _ = x_test.shape
        errors = []
        y_gx = -torch.sign(torch.mv(x_test, w_test))
        for i in range(N):
            if (y_test[i] != y_gx[i]):
                errors.append(i)
        return errors

    def loss(self):
        return torch.sum(torch.log(1+torch.exp(self.y.reshape(-1, 1)*torch.mv(self.x, self.w).reshape(-1, 1))), dim=0)/len(self.x)

    def predict(self, x_test=None):
        if x_test == None:
            x_test = self.x
        #print(x_test.shape, self.w.shape)
        return torch.sigmoid(torch.mv(x_test, self.w))

    def gd_train(self, a=0.5, epoch_num=1000, batch=100):
        N = len(self.x)
        loss = [self.loss()]
        for i in range(epoch_num):
            for j in range(N//batch+1):
                x_batch = self.x[batch*j:min(batch*(j+1), N), :]
                y_batch = self.y[batch*j:min(batch*(j+1), N)]
                gd_w = (-torch.sigmoid(-y_batch.reshape(-1, 1)*torch.mv(x_batch, self.w).reshape(-1, 1)
                                       ).reshape(1, -1)@(-y_batch.reshape(-1, 1)*x_batch)).reshape(-1)
                self.w -= a * gd_w
                accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
            loss.append(self.loss().nan_to_num())
            # if torch.norm(gd_w) < 1e-5:
            #     break
        pplt.figure(1)
        print(loss)
        pplt.plot(range(len(loss)), loss, '-')
        pplt.figure(2)
        self.draw()
        accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
        return self.w, accuracy

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
        N, _ = x_test.shape
        x_ex = torch.cat((torch.ones(N).reshape(N, 1), x_test), dim=1)
        err_index = self.errors(self.w, x_ex, y_test)
        err = len(err_index)
        accuracy = 1-err/N
        print(self.predict(x_ex))
        print("test: accuracy: ", float(accuracy))
        pplt.title("test")
        self.draw(x_ex, y_test)
