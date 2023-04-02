from SVM import SVM, Kernel_SVM
import torch

# test1
'''x1 = torch.randn(200, 2)+torch.tensor([-5, 0])
x2 = torch.randn(200, 2)+torch.tensor([0, 5])
train_num = int(200*0.8)
test_num = 200-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
x_train = torch.cat((x1_train, x2_train), dim=0)
y_train = torch.cat((torch.ones(train_num), -torch.ones(train_num)), dim=0)
x_test = torch.cat((x1_test, x2_test), dim=0)
y_test = torch.cat((torch.ones(test_num), -torch.ones(test_num)), dim=0)'''

'''sample = SVM(x_train, y_train)
sample.Dual_SVM()
sample.test(x_test, y_test)'''

'''sample = Kernel_SVM(x_train, y_train)
sample.train(gama = 0.1)
sample.test(x_test, y_test)'''

#test2
'''x1 = torch.randn(200, 2)+torch.tensor([3, 0])
x2 = torch.randn(200, 2)+torch.tensor([0, 3])
train_num = int(200*0.8)
test_num = 200-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
x_train = torch.cat((x1_train, x2_train), dim=0)
y_train = torch.cat((torch.ones(train_num), -torch.ones(train_num)), dim=0)
x_test = torch.cat((x1_test, x2_test), dim=0)
y_test = torch.cat((torch.ones(test_num), -torch.ones(test_num)), dim=0)'''

'''sample = SVM(x_train, y_train)
sample.Primal_SVM()
sample.test(x_test, y_test)

sample = SVM(x_train, y_train)
sample.Dual_SVM()
sample.test(x_test, y_test)'''

'''sample = Kernel_SVM(x_train, y_train)
sample.train(method='4')
sample.test(x_test, y_test)

sample = Kernel_SVM(x_train, y_train)
sample.train(gama=0.1)
sample.test(x_test, y_test)'''

#test3
china_sea=torch.tensor([[119.28,26.08],         # 福州
                        [121.31,25.03],         # 台北
                        
                        [121.47,31.23],         # 上海
                        [118.06,24.27],         # 厦门
                        [121.46,39.04],         # 大连
                        [122.10,37.50],         # 威海
                        [124.23,40.07]])        # 丹东

j_sea=torch.tensor([    [129.87,32.75],         # 长崎
                        [130.33,31.36],         # 鹿儿岛
                        [131.42,31.91],         # 宫崎
                        [130.24,33.35],         # 福冈
                        [133.33,15.43],         # 鸟取
                        [138.38,34.98],         # 静冈
                        [140.47,36.37]])        # 水户   

china_land=torch.tensor([[113.53,29.58],        # 武汉
                        [104.06,30.67],         # 成都
                        [116.25,39.54]])        # 北京

j_land=torch.tensor([   [136.54,35.10],         # 名古屋
                        [132.27,34.24],         # 广岛
                        [139.46,35.42]])        # 东京

Diaoyu = torch.tensor([123.28,25.45]).reshape(1, 2)
# q1
train_x = torch.cat((china_sea, j_sea), dim=0)
train_y = torch.cat((torch.ones(china_sea.shape[0]), -torch.ones(j_sea.shape[0])), dim=0)
model = SVM(train_x, train_y)
model.Dual_SVM()
print(model.predict(Diaoyu))
#q2
train_x = torch.cat((china_sea, china_land, j_sea, j_land), dim=0)
train_y = torch.cat((torch.ones(china_sea.shape[0]+china_land.shape[0]), -torch.ones(j_sea.shape[0]+j_land.shape[0])), dim=0)
model = SVM(train_x, train_y)
model.Dual_SVM()
print(model.predict(Diaoyu))