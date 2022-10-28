# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
""""
用LSTM分类scv数据
数据集是实验室数据集
accuracy =  1代87%
70%train
30%test
"""""
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

torch.manual_seed(1)    # reproducible
# Hyper Parameters
EPOCH = 1#30           # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.01              # learning rate


#读csv函数(读数据)
def R_xcsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数

    #读取的date是字符串，将其转化为数值
    for i in range(len(date)):
        date[i] = list(map(float, date[i]))
    # for i in range(len(date)):
    #     date[i] = np.array(date[i]).reshape(8, 8)#将列表的元素转化为8 x 8
    # date = np.array(date, dtype=float)  # trainX为待转化的列表
    return date   #返回的数据是浮点型存在list中


#读标签函数
def R_ycsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数

    for i in range(len(date)):#将读取的字符串转化为数值
        date[i] = list(map(int, date[i]))
    # date = np.array(date, dtype=float)  # trainX为待转化的列表
    return date

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

#训练数据的构建 读取数据集中的所有数据
Mat1=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d0.csv')
Mat2=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d1.csv')
Mat3=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d2.csv')
Mat4=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d3.csv')
Mat5=R_xcsv(r'C:\Users\86182\Desktop\Dataset\data\d4.csv')
totalMat=np.concatenate((Mat1,Mat2,Mat3,Mat4,Mat5),axis=0)#垂直组合numpy
# trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#垂直组合numpy

Labels1= [0] * len(Mat1)
Labels2= [1] * len(Mat2)
Labels3= [2] * len(Mat3)
Labels4= [3] * len(Mat4)
Labels5= [4] * len(Mat5)
total_Labels=np.concatenate((Labels1,Labels2,Labels3,Labels4,Labels5),axis=0)#垂直组合numpy


# 打乱索引
#得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
totalMat=np.array(totalMat)
total_Labels=np.array(total_Labels).reshape(len(total_Labels),) # 将标签修改成一维


index = [i for i in range(len(totalMat))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
totalMat = totalMat[index]
total_Labels = total_Labels[index]


# # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
# trainX=np.array(trainingMat)
# trainX = trainX.reshape((len(trainX),1,8,8))
#
# # print(trainX,train_hwLabels)
# a = trainX
# b = train_hwLabels

# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
totaldata =np.array(totalMat)
totaldata = totaldata.reshape((len(totaldata),1,8,8))
totallable = total_Labels

# 划分训练集和测试集数据70%训练集，30%测试集
trainX = totaldata[0:151200]
train_Labels = totallable[0:151200]

testX = totaldata[151201:]
test_Labels = totallable[151201:]

##封装训练集数据：
#将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
train_x = torch.tensor(trainX).type(torch.FloatTensor)
train_y = torch.tensor(train_Labels, dtype=torch.long)#csv读取int转化为long

 #将标签和输入数据用自定义函数封装
input_data = GetLoader(train_x , train_y)
train_data= DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)


#测试集数据
test_x = torch.tensor(testX).type(torch.FloatTensor)
test_y = torch.tensor(test_Labels, dtype=torch.long)#csv读取int转化为long


# #建立cnn网络模型  87%
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         # input shape (1, 8, 8)维度 图片的宽和高
#             nn.Conv2d(
#                 in_channels=1,              # input height（输入的通道数） （16， 8， 8）
#                 out_channels=16,            # n_filters输出的通道数，将原来的一维变成16维
#                 kernel_size=3,              # filter size
#                 stride=1,                   # filter movement/step
#                 padding=1,                  # if want same width and length of this image after Conv2d,
#                                             # padding=(kernel_size-1)/2 if stride=1
#             ),                              # output shape (16, 8, 8)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 4, 4)
#         )
#         self.conv2 = nn.Sequential(         # input shape (16, 4, 4)
#             # nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 4, 4)
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(2),                # output shape (32, 2, 2)
#         )
#         #self.out = nn.Linear(32 * 8 * 8, 5)   # fully connected layer, output 10 classes
#         self.out = nn.Linear(32 * 2 * 2, 5)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)                  #(batch_size, 32 * 8 * 8)
#         x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 8 * 8)
#                                             # x.size(0)表示的是batch
#         output = self.out(x)
#         return output, x    # return x for visualization



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
            nn.Conv2d(16, 16, 5, 1, 2),     # output shape (32, 4, 4)
            # nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 2, 2)
        )
        self.out = nn.Linear(16 * 2 * 2, 5)   # fully connected layer, output 5 classes
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


# training and testing
#定义模型保存的绝对路径
modelfilepath = r'C:\Users\86182\Desktop\LearnText\pytorch\test\practice\CNN\Reload\CNN.test1.pkl'
acc = 0
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_data):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)#拿测试集去验证
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # print(pred_y,type(pred_y),len(pred_y))#预测的标签有1934个出现的原因主要是因为数据出错了
            # print(test_y.data.numpy(),type(test_y.data.numpy()),len(test_y.data.numpy()))#实际的标签有946个
            # print(float(test_y.size(0)))
            # print(np.sum(pred_y==test_y.data.numpy()))
            # print(float((pred_y == test_y.data.numpy()).astype(int).sum()))
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            # 保存精度最高的模型
            if acc <= accuracy:
                torch.save(lstm, modelfilepath)  # save entire net保存整个神经网络，‘   ’保存的形式
                acc = accuracy
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# print 10 predictions from test data
net = torch.load(modelfilepath)#加载模型
test_output, _ = net(test_x[:50])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:50].numpy(), 'real number')
# #






































# # training and testing
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(train_data):        # gives batch data
#         # b_x = b_x.view(-1, 8, 8)              # reshape x to (batch, time_step, input_size)
#                                                 #rnn接收数据的形式
#
#         output = cnn(b_x)[0]  # cnn output
#         loss = loss_func(output, b_y)  # cross entropy loss
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients
#
#         if step % 100 == 0:
#             test_output, last_layer = cnn(test_x)  # 拿测试集去验证
#                                                 # (samples, time_step, input_size)
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
#
# # print 10 predictions from test data
# test_output, _ = cnn(test_x[:50])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, 'prediction number')
# print(test_y[:50].numpy(), 'real number')


