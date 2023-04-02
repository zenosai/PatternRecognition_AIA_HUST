import torch.optim as optim
import torchvision.datasets as dataset
import matplotlib.pyplot as pplt
import torchvision.transforms as T
from torch.utils.data import DataLoader
from Model import LeNet, train, show_result, show_mnist_sample

pplt.rcParams['image.cmap'] = 'gray'

def load_mnist(path='./mnist', batch_size=64):
    transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.5), (0.5)) # mnist默认灰度图像, 就1个
              ])
    
    mnist_train = dataset.MNIST(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(mnist_train, batch_size, shuffle = True)
    
    mnist_test = dataset.MNIST(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size, shuffle = True)
    
    return loader_train, loader_test


if __name__ == '__main__':
    loader_train, loader_test = load_mnist(batch_size=256)  # 64效果比256好很多
    _, H, W = loader_train.dataset.data.shape

    learning_rate = 0.2
    momentum = 0.5

    model = LeNet(1, H, W)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                          momentum=momentum, nesterov=True)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=10)
    show_result(acc_history, acc_test_history, loss_history, iter_history)
    show_mnist_sample(model, loader_test)