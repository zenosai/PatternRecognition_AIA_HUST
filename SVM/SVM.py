import torch
import matplotlib.pyplot as pplt
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, x, y):
        N, dim = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1).double()
        self.y = y.double()
        self.w = torch.zeros(dim+1).double()

    def Primal_SVM(self):
        P = torch.eye(self.x.shape[1])
        P[0][0] = 0
        P = matrix(P.double().tolist())
        q = torch.zeros(self.x.shape[1])
        q = matrix(q.double().tolist())
        G = matrix((-self.y.reshape(-1, 1)*self.x).t().double().tolist())
        h = -torch.ones(self.x.shape[0])
        h = matrix(h.double().tolist())

        w = list(solvers.qp(P, q, G, h)['x'])
        self.w = torch.tensor(w).double()

        accuracy = 1-len(self.errors(self.w))/len(self.x)
        print(self.w, accuracy)
        self.draw()
        return self.w, accuracy

    def Dual_SVM(self):
        x = self.x.split([1, 2], 1)[1]
        y = self.y.reshape(-1, 1)

        P = matrix((y.t()*(x@x.t())*y).t().double().tolist())
        q = -torch.ones(x.shape[0])
        q = matrix(q.double().tolist())
        G = -torch.eye(len(x))
        G = matrix(G.double().tolist())
        h = -torch.zeros(x.shape[0])
        h = matrix(h.double().tolist())
        A = matrix(y.double().tolist())
        B = matrix([0.])

        alpha = list(solvers.qp(P, q, G, h, A, B)['x'])
        self.alpha = torch.tensor(alpha)

        w = ((self.alpha*self.y).t()@x).reshape(-1)

        self.i_sv = [i for i in range(len(self.alpha)) if self.alpha[i] > 1e-5]
        b = self.y[self.i_sv[0]] - torch.dot(x[self.i_sv[0]], w)

        self.w = torch.cat((torch.tensor([b]), w), dim=0)

        accuracy = 1-len(self.errors(self.w))/len(self.x)
        print(self.w, accuracy)
        self.draw()
        return self.w, accuracy

    def errors(self, w_test, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
        N, _ = x_test.shape
        y_gx = torch.sign(torch.mv(x_test, w_test))
        return [i for i in range(N) if y_test[i] != y_gx[i]]

    def draw(self, x_test=None, y_test=None):
        test = True
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
            test = False
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

        if not test:
            for i in self.i_sv:
                pplt.plot(x_test[i, 1], x_test[i, 2],
                          'go' if y[i] == 1 else 'gx')
                y_line_sv = -1*x_line * \
                    self.w[1]/self.w[2] + \
                    (self.w[1]*x_test[i, 1]+self.w[2]*x_test[i, 2])/self.w[2]
                pplt.plot(x_line, y_line_sv, 'y-.')

        pplt.show()

    def test(self, x_test, y_test):
        N, _ = x_test.shape
        x_ex = torch.cat((torch.ones(N).reshape(N, 1), x_test), dim=1).double()
        err_index = self.errors(self.w, x_ex, y_test.double())
        err = len(err_index)
        rate = 1-err/N
        print("test: accuracy: ", float(rate))
        pplt.title("test")
        self.draw(x_ex, y_test)

    def predict(self, x_test=None):
        if x_test == None:
            x_test = self.x
        else:
            N, _ = x_test.shape
            x_test = torch.cat(
                (torch.ones(N).reshape(N, 1), x_test), dim=1).double()
        pplt.plot(x_test[:, 1], x_test[:, 2], 'r*')
        self.draw()
        return torch.sign(torch.mv(x_test, self.w))


class Kernel_SVM:
    def __init__(self, x, y):
        self.x = x.double()
        self.y = y.double()
        self.w = torch.zeros(x.shape[1]+1)

    def K(self, x1, x2, method='gauss', delta=1, gama=1):
        assert ((method in ['gauss', '2', '4']) or (
            isinstance(method, int) and method > 0))
        if method == 'gauss':
            xx1 = (x1*x1).sum(dim=1)
            xx2 = (x2*x2).sum(dim=1)
            z = 2*(x1@x2.t())-xx1.reshape(-1, 1)-xx2.reshape(1, -1)
            return torch.exp(gama*z)
        elif method == '2':
            return 1+x1@x2.t()+(x1@x2.t())**2
        elif method == '4':
            return 1+x1@x2.t()+0.01*(x1@x2.t())**4
        else:
            return (delta*(x1@x2.t())+gama)**method

    def train(self, method='gauss', delta=1, gama=1):
        self.method, self.delta, self.gama = method, delta, gama
        z = self.K(self.x, self.x, method, delta, gama)
        y = self.y.reshape(-1, 1)

        P = matrix((y.t()*z*y).t().double().tolist())
        q = -torch.ones(len(z))
        q = matrix(q.double().tolist())
        G = -torch.eye(len(z))
        G = matrix(G.double().tolist())
        h = -torch.zeros(len(z))
        h = matrix(h.double().tolist())
        A = matrix(y.double().tolist())
        B = matrix([0.])

        alpha = list(solvers.qp(P, q, G, h, A, B)['x'])
        self.alpha = torch.tensor(alpha).reshape(-1, 1)

        self.i_sv = [i for i in range(len(self.alpha)) if self.alpha[i] > 1e-5]
        self.a_sv = self.alpha[self.i_sv]
        self.x_sv = self.x[self.i_sv]
        self.y_sv = y[self.i_sv]
        self.b = self.y_sv[0] - torch.dot((self.a_sv*self.y_sv).reshape(-1), self.K(
            self.x_sv, self.x_sv[0].reshape(1, -1), method, delta, gama).reshape(-1))
        print(self.b)
        accuracy = 1-len(self.errors())/len(self.x)
        print(accuracy)
        self.draw()
        return accuracy

    def errors(self, x_test=None, y_test=None):
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y.reshape(-1, 1)
        else:
            y_test = y_test.reshape(-1, 1)
        N, _ = x_test.shape
        y_gx = torch.sign(self.b + torch.mv(self.K(self.x_sv, x_test, self.method, self.delta,
                          self.gama).t(), (self.a_sv*self.y_sv).reshape(-1)))
        return [i for i in range(N) if y_test[i] != y_gx[i]]

    def draw(self, x_test=None, y_test=None):
        test = True
        if x_test == None or y_test == None:
            x_test, y_test = self.x, self.y
            test = False
        y = y_test
        err_indexes = self.errors(x_test, y_test)

        x_pos_right = x_test[[index for index in range(
            len(x_test)) if index not in err_indexes and y[index] == 1]]
        x_neg_right = x_test[[index for index in range(
            len(x_test)) if index not in err_indexes and y[index] == -1]]

        x_pos_false = x_test[[index for index in err_indexes if y[index] == 1]]
        x_neg_false = x_test[[
            index for index in err_indexes if y[index] == -1]]

        pplt.plot(x_pos_right[:, 0], x_pos_right[:, 1], 'bo')
        pplt.plot(x_neg_right[:, 0], x_neg_right[:, 1], 'bx')

        pplt.plot(x_pos_false[:, 0], x_pos_false[:, 1], 'ro')
        pplt.plot(x_neg_false[:, 0], x_neg_false[:, 1], 'rx')

        if not test:
            for i in self.i_sv:
                pplt.plot(x_test[i, 0], x_test[i, 1],
                          'go' if y[i] == 1 else 'gx')
        
        xx = torch.linspace(torch.min(x_test[:,0])-0.5, torch.max(x_test[:,0])+0.5, 25)
        yy = torch.linspace(torch.min(x_test[:,1])-0.5, torch.max(x_test[:,1])+0.5, 25)
        X,Y = torch.meshgrid(xx, yy)
        xxx = torch.transpose(torch.transpose(torch.stack((X, Y),dim=0),0,2),1,0).reshape(-1,2).double()
        f = ((self.alpha.reshape(1,-1)*self.y.reshape(1,-1))@self.K(self.x, xxx, self.method, self.delta, self.gama)).reshape(25, 25)+self.b
        contour = pplt.contourf(X, Y, f,levels=10, alpha=.75, cmap='coolwarm')
        CS = pplt.contour(X, Y, f, linewidths=1, linestyles='dashed', levels=[-1, 0, 1], colors='k')
        pplt.clabel(CS, inline=True)
        pplt.colorbar(contour)

        pplt.show()

    def test(self, x_test, y_test):
        N, _ = x_test.shape
        err_index = self.errors(x_test.double(), y_test.double())
        err = len(err_index)
        rate = 1-err/N
        print("test: accuracy: ", float(rate))
        pplt.title("test")
        self.draw(x_test.double(), y_test.double())