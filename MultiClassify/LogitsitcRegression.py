import torch


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
        y_gx = torch.sign(torch.mv(x_test, w_test))
        for i in range(N):
            if (y_test[i] != y_gx[i]):
                errors.append(i)
        return errors

    def loss(self):
        return torch.log(1+torch.exp(-torch.mm(self.y.reshape(1, -1),
                  torch.mv(self.x, self.w).reshape(-1, 1))))

    def predict(self, x_test=None):
        if x_test == None:
            x_test = self.x
        return torch.sigmoid(torch.mv(x_test, self.w))

    def gd_train(self, a=0.5, epoch_num=1000, batch=100):
        N = len(self.x)
        loss = []
        for i in range(epoch_num):
            for j in range(N//batch+1):
                x_batch = self.x[batch*j:min(batch*(j+1), N), :]
                y_batch = self.y[batch*j:min(batch*(j+1), N)]
                gd_w = (-torch.sigmoid(-torch.mm(y_batch.reshape(1, -1), torch.mv(x_batch,
                        self.w).reshape(-1, 1)))*torch.mv(x_batch.t(), y_batch)).reshape(-1)
                print(self.w, gd_w)
                self.w -= a * gd_w
                accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
            loss.append(self.loss())
        accuracy = 1-len(self.errors(self.w))/self.x.shape[0]
        return self.w, accuracy
