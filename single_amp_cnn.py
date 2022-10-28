# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
"""
###划分后的数据单个传感器的幅值###
20帧
"""


import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from view_result import plot_result
import matplotlib.pyplot as plt

from scipy import signal



BATCH_SIZE = 32
LR = 0.01
EPOCH = 1000

#绘制epoch和loss曲线
def drawlearnrate(train_loss):
    # 绘制训练 & 验证的损失值
    plt.plot(train_loss)
    # plt.plot(val_loss)
    plt.title('train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


def low(data,w):
    b, a = signal.butter(8, w, 'lowpass')  #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) #data为要过滤的信号
    return filtedData

#构建pytorch行驶本数据集函数
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


amp0_train = np.load('./dataset/split_train_val/data/train/csi_a/csi_a1_00_train.npy')#(6183, 62)
amp1_train = np.load('./dataset/split_train_val/data/train/csi_a/csi_a1_11_train.npy')#(3534, 62)
amp2_train = np.load('./dataset/split_train_val/data/train/csi_a/csi_a1_22_train.npy')#(3830, 62)
amp3_train = np.load('./dataset/split_train_val/data/train/csi_a/csi_a1_33_train.npy')#(2225, 62)


amp0_test = np.load('./dataset/split_train_val/data/test/csi_a/csi_a1_00_test.npy')#(1455, 62)
amp1_test = np.load('./dataset/split_train_val/data/test/csi_a/csi_a1_11_test.npy')#(763, 62)
amp2_test = np.load('./dataset/split_train_val/data/test/csi_a/csi_a1_22_test.npy')#(964, 62)
amp3_test = np.load('./dataset/split_train_val/data/test/csi_a/csi_a1_33_test.npy')#(617, 62)

amp0_train = amp0_train[0:6180]
amp0_train = low(amp0_train,0.2)
amp0_train_label = [0] *309

amp1_train = amp1_train[0:3520]
amp1_train = low(amp1_train,0.2)
amp1_train_label = [1] *176

amp2_train = amp2_train[0:3820]
amp2_train = low(amp2_train,0.2)
amp2_train_label = [2] *191

amp3_train = amp3_train[0:2220]
amp3_train = low(amp3_train,0.2)
amp3_train_label = [3] *111


amp0_test = amp0_test[0:1440]
amp0_test = low(amp0_test,0.2)
amp0_test_label = [0] *72

amp1_test = amp1_test[0:760]
amp1_test = low(amp1_test,0.2)
amp1_test_label = [1] *38

amp2_test = amp2_test[0:960]
amp2_test = low(amp2_test,0.2)
amp2_test_label = [2] *48

amp3_test = amp3_test[0:600]
amp3_test = low(amp3_test,0.2)
amp3_test_label = [3] *30

train_data = np.concatenate((amp0_train,amp1_train,amp2_train,amp3_train),axis = 0)
train_label = np.concatenate((amp0_train_label,amp1_train_label,amp2_train_label,amp3_train_label),axis = 0)

test_data = np.concatenate((amp0_test,amp1_test,amp2_test,amp3_test),axis = 0)
test_label = np.concatenate((amp0_test_label,amp1_test_label,amp2_test_label,amp3_test_label),axis = 0)

trainX = train_data.reshape((-1,1,20,62))
testX = test_data.reshape((-1,1,20,62))

index = [i for i in range(len(trainX))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
trainX = trainX[index]
train_label = train_label[index]

index = [i for i in range(len(testX))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
testX= testX[index]
test_label = test_label[index]


train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(train_label, dtype=torch.long)#csv读取int转化为long

#  #将标签和输入数据用自定义函数封装
input_train_data = GetLoader(train_x , train_y)
train_data_loader = DataLoader(input_train_data, batch_size=BATCH_SIZE, shuffle=True)

# #测试集数据
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_label, dtype=torch.long)#csv读取int转化为long

input_test_data = GetLoader(test_x , test_y)
test_data_loader = DataLoader(input_test_data, batch_size=BATCH_SIZE, shuffle=True)

# print(amp0_test.shape)
# print(amp1_test.shape)
# print(amp2_test.shape)
# print(amp3_test.shape)



#建立cnn网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 8, 8)维度 图片的宽和高
            nn.Conv2d(
                in_channels=1,              # input height（输入的通道数） （1， 8， 8）
                out_channels=16,            # n_filters输出的通道数，将原来的一维变成16维（16， 8， 8）
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d,
                                            # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 8, 8)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 4, 4)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 4, 4)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 4, 4)
            # nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 2, 2)
        )
        self.out = nn.Linear(32 * 15 * 5, 4)   # fully connected layer, output 5 classes
        #self.out = nn.Linear(32 * 2 * 2, 5)  # fully connected layer, output 5 classes（0.88）
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                  #(batch_size, 32 * 8 * 8)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
                                            # x.size(0)表示的是batch
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
print(cnn)  # net architecture


#优化器和损失函数的选择
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


# # training and testing
# #定义模型保存的绝对路径
modelfilepath = './model/lowbutter_data_split_CNN_single_amp_500_epoch1.pkl'
acc = 0
keep_loss = []
middle_loss = 10000
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if loss.data.numpy()<middle_loss:
            keep_loss.append(loss.data.numpy())
            middle_loss = loss.data.numpy()
        else:
            keep_loss.append(middle_loss)

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)#拿测试集去验证
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            # 保存精度最高的模型
            if acc <= accuracy:
                torch.save(cnn, modelfilepath)  # save entire net保存整个神经网络，‘   ’保存的形式
                acc = accuracy
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

drawlearnrate(keep_loss)#画train——loss 和 epoch
# print 10 predictions from test data
net = torch.load(modelfilepath)#加载模型
test_output, _ = net(test_x[:50])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:50].numpy(), 'real number')
# #

plot_result(test_x,test_y,net)
