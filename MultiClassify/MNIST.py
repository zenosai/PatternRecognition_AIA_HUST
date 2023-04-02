import torch
import torch.optim as optim
import matplotlib.pyplot as pplt
import torchvision.datasets as ds
import torchvision.transforms as T
from torch.utils.data import DataLoader
import math

from Softmax import Softnn, train


def load_mnist(path='./mnist', batch_size=64):
    transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.5), (0.5)) # mnist默认灰度图像, 就1个
              ])
    
    mnist_train = ds.MNIST(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(mnist_train, batch_size, shuffle = True)
    
    mnist_test = ds.MNIST(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size, shuffle = True)
    
    return loader_train, loader_test


def show_mnist_sample(model, loader, batch_size=64, num=10, row=5):
    if loader.dataset.train:
        print('show random sample on train set')
    else:
        print('show random sample on test set')
    full_num = batch_size
    sample = list(torch.utils.data.WeightedRandomSampler(torch.ones(full_num)/full_num,num_samples=num, replacement=False))
    examples = enumerate(loader)
    _, (example_data, example_targets) = next(examples)
    fig = pplt.figure(figsize=(8, 6))
    for i in range(num):
        pplt.subplot(math.ceil(num/row), row, i+1)
        pplt.imshow(example_data[sample[i]][0], cmap='gray', interpolation='none')
        pplt.title("pdct: {}, actl: {}".format((model(example_data)[sample[i]]).argmax(), example_targets[sample[i]]))
        pplt.xticks([])
        pplt.yticks([])
    pplt.show()

def show_result(acc_history, acc_test_history, loss_history, iter_history):
    fig = pplt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(iter_history, acc_history, 'y-', label='train accuracy')
    ax.plot(iter_history, acc_test_history, 'g-', label='test accuracy')
    ax2.plot(iter_history, loss_history, 'c-', label='loss')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.set_ylabel(r"accuracy")
    ax2.set_ylabel(r"loss")
    pplt.show()


if __name__ == '__main__':
    loader_train, loader_test = load_mnist(batch_size=256)

    dim_in = 784
    C = 10
    learning_rate = 0.1
    momentum = 0.5

    model = Softnn(dim_in, C)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=10)
    show_result(acc_history, acc_test_history, loss_history, iter_history)
    show_mnist_sample(model, loader_test)