import numpy as np
import math
import matplotlib.pyplot as pplt

def f(x):
    return x * np.cos(0.25 * math.pi * x)

def grad(x0, method='gd', beta1=0.9, beta2=0.999, lr=0.4, iter_max=10):
    x = x0
    x_history = [x]
    v = 0
    m = 0
    tmp = 0
    for i in range(iter_max):
        gdx = math.cos(0.25 * math.pi * x) - math.sin(0.25 * math.pi * x) * 0.25 * math.pi * x
        if method=='gd':
            x -= lr*gdx
        elif method=='Momentum': 
            m = beta1*m-gdx
            x += lr*m
        elif method=='RMS':       # NAN(梯度为-, 不能开方), 加abs
            v = math.sqrt(beta1*v**2+(1-beta1)*gdx**2)
            x -= lr*gdx/(v+1e-6)
        elif method=='Ada':
            tmp = tmp + gdx**2
            v = math.sqrt(tmp/(i+1))
            x -= lr*gdx/(v+1e-6)
            
        elif method=='Adam':
            m = beta1*m-gdx
            v = math.sqrt(beta2*v**2+(1-beta2)*gdx**2)
            x += lr*m/(v+1e-6)
        x_history.append(x)
    return x_history

x0 = 4
x_ = np.linspace(3.9,4.9)
y_ = f(x_)
pplt.figure(1)
pplt.title('GD')
pplt.plot(x_,y_,'g.-.')
x_history = np.array(grad(x0, 'gd', iter_max=50))
y_his = f(x_history)
pplt.plot(x_history,y_his,'r.-.')

x0 = 4
x_ = np.linspace(3.9,4.9)
y_ = f(x_)
pplt.figure(2)
pplt.title('Adagrad')
pplt.plot(x_,y_,'g.-.')
x_history = np.array(grad(x0, 'Ada', iter_max=50))
y_his = f(x_history)
pplt.plot(x_history,y_his,'r.-.')

x0 = 4
x_ = np.linspace(3.9,5.4)
y_ = f(x_)
pplt.figure(3)
pplt.title('RMSProp')
pplt.plot(x_,y_,'g.-.')
x_history = np.array(grad(x0, 'RMS', iter_max=50))
y_his = f(x_history)
pplt.plot(x_history,y_his,'r.-.')

x0 = 4
x_ = np.linspace(3.9,4.8)
y_ = f(x_)
pplt.figure(4)
pplt.title('Momentum')
pplt.plot(x_,y_,'g.-.')
x_history = np.array(grad(x0, 'Momentum', iter_max=50))
y_his = f(x_history)
pplt.plot(x_history,y_his,'r.-.')

x0 = 4
x_ = np.linspace(-80,80,500)
y_ = f(x_)
pplt.figure(5)
pplt.title('Adam')
pplt.plot(x_,y_,'g.-.')
x_history = np.array(grad(x0, 'Adam', beta1=0.99, iter_max=50))
y_his = f(x_history)
pplt.plot(x_history,y_his,'r.-.')


pplt.show()
