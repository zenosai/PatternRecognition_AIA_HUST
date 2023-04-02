import torch
import matplotlib.pyplot as pplt


class FLD():
    def __init__(self, x1, x2):
        self.x1, self.x2 = x1, x2
        self.N1, dim = self.x1.shape
        self.N2, _ = self.x2.shape
        self.w = torch.zeros(dim+1)

    def FLDA(self):
        miu1 = self.x1.mean(dim=0)
        miu2 = self.x2.mean(dim=0)
        Sw = torch.mm((self.x1-miu1).t(), self.x1-miu1)/self.N1 + \
            torch.mm((self.x2-miu2).t(), self.x2-miu2)/self.N2
        self.w = torch.mm(torch.inverse(
            Sw), (miu1-miu2).reshape(-1, 1)).reshape(-1)
        miu = (miu1+miu2)/2
        self.s = torch.dot(self.w, miu)
        self.y1 = 1 if torch.dot(self.w, miu1) > self.s else -1
        self.y2 = 1 if torch.dot(self.w, miu2) > self.s else -1
        err_indexes1, err_indexes2 = self.errors(self.x1, self.x2)
        accuracy = 1-(len(err_indexes1)+len(err_indexes2))/(len(self.x1)+len(self.x2))
        print('train accuracy:',accuracy)
        print('w:',self.w,'y0:',self.s)
        pplt.title('FLDA train')
        self.draw()
        return [self.w, self.s]

    def errors(self, x1_test, x2_test):
        if x1_test == None or x2_test == None:
            x1_test, x2_test = self.x1, self.x2
        errors1 = []
        y_gx1 = torch.sign(torch.mv(x1_test, self.w)-self.s)
        for i in range(len(x1_test)):
            if (y_gx1[i] != 1):
                errors1.append(i)
        errors2 = []
        y_gx2 = torch.sign(torch.mv(x2_test, self.w)-self.s)
        for i in range(len(x2_test)):
            if (y_gx2[i] != -1):
                errors2.append(i)
        return errors1, errors2

    def draw(self, x1_test=None, x2_test=None):
        if x1_test == None or x2_test == None:
            x1_test, x2_test = self.x1, self.x2
        err_indexes1, err_indexes2 = self.errors(x1_test, x2_test)

        x1_right = x1_test[[index for index in range(
            len(x1_test)) if index not in err_indexes1]]
        x2_right = x2_test[[index for index in range(
            len(x2_test)) if index not in err_indexes2]]

        x1_false = x1_test[[index for index in range(
            len(x1_test)) if index in err_indexes1]]
        x2_false = x2_test[[index for index in range(
            len(x2_test)) if index in err_indexes2]]

        pplt.plot(x1_right[:, 0], x1_right[:, 1], 'bo')
        pplt.plot(x2_right[:, 0], x2_right[:, 1], 'bx')

        pplt.plot(x1_false[:, 0], x1_false[:, 1], 'ro')
        pplt.plot(x2_false[:, 0], x2_false[:, 1], 'rx')

        x_line = torch.tensor([min(min(x1_test[:, 0]), min(x2_test[:, 0])), max(
            max(x1_test[:, 0]), max(x2_test[:, 0]))])
        y_line = -1*x_line*self.w[0]/self.w[1]+self.s/self.w[1]
        pplt.plot(x_line, y_line, '-')
        y_line_w = x_line*self.w[1]/self.w[0]
        pplt.plot(x_line, y_line_w, '-.')
        pplt.axis('equal')
        pplt.show()

    def test(self, x1_test, x2_test):
        err_indexes1, err_indexes2 = self.errors(x1_test, x2_test)
        accuracy = 1-(len(err_indexes1)+len(err_indexes2))/(len(x1_test)+len(x2_test))
        print('train accuracy:',accuracy)
        pplt.title('test')
        self.draw(x1_test, x2_test)
