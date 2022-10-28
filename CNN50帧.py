# -*- coding: utf-8 -*-
"""
Created on Fri May 13 08:36:30 2022

@author: wzw
"""

import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPool2D
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from keras.models import load_model
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler # 数据预处理
from scipy import signal
import joblib   ###存训练模型

def save_model(model, filepath):
	joblib.dump(model, filename=filepath)       #保存模型

# def load_model(filepath):
# 	model = joblib.load(filepath)
# 	return model

#*************随机种子，模型复现*******#
seed = 42
np.random.seed(seed)


#********滤波*******#
def low(data,w):
    b, a = signal.butter(8, w, 'lowpass')  #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) #data为要过滤的信号
    return filtedData

# amp0 = np.load(r'D:\Desktop\group\group_member\tyx\dataset\bedroom_empty_ta1.npy')
# amp1 = np.load(r'D:\Desktop\group\group_member\tyx\dataset\bedroom_sitting_ta1.npy')
# amp2 = np.load(r'D:\Desktop\group\group_member\tyx\dataset\toilet_empty_ta3.npy')
# amp3 = np.load(r'D:\Desktop\group\group_member\tyx\dataset\toilet_sitting_ta1.npy')


amp0 = np.load('./dataset/bedroom_empty_ta1.npy')
amp1 = np.load('./dataset/bedroom_sitting_ta1.npy')
amp2 = np.load('./dataset/toilet_empty_ta3.npy')
amp3 = np.load('./dataset/toilet_sitting_ta1.npy')



  #*********五十帧送入*******#
amp0_train = amp0[0:23000]
amp0_train = low(amp0_train,w=0.21)
amp0_train_label = [0] *460

amp1_train = amp1[0:23000]
amp1_train = low(amp1_train,w=0.2)
amp1_train_label = [1] *460

amp2_train = amp2[0:22000]
amp2_train = low(amp2_train,w=0.3)
amp2_train_label = [2] *440



amp3_train = amp3[0:23000]
amp3_train = low(amp3_train,w=0.15)
amp3_train_label = [3] *460



train_data = np.concatenate((amp0_train,amp1_train,amp2_train,amp3_train),axis = 0)
train_label = np.concatenate((amp0_train_label,amp1_train_label,amp2_train_label,amp3_train_label),axis = 0)


y1 =train_label
y =np.array(y1)

x = train_data.reshape(-1,62*50)

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    normDataSet = np.zeros(dataSet.shape)
    normDataSet = (dataSet - minVals)/(maxVals - minVals)
    return normDataSet

# train_data = autoNorm(train_data)
# train_data,test_data,y_train,y_test=sklearn.model_selection.train_test_split(train_data,train_label,test_size=0.2,random_state=0)
##***********y_train,y_test分别前为训练集与测试集标签******##
x=autoNorm(x)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=0)


Labels = []
Labels.append(to_categorical(y_train, 4))
trainY1 = np.array(Labels)
trainY=trainY1.reshape(trainY1.shape[1], trainY1.shape[2])
# trainY=np.array(Labels)

trainX = x_train.reshape((x_train.shape[0],50,62, 1))



inp=Input(shape=(50,62,1))

x = Conv2D(30, (2, 2), padding='valid', strides=(1, 1), activation='relu', name='conv-1')(inp)
x = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', name='pool-1')(x)
# print(x.shape)
x = Conv2D(30, (2, 2), padding='valid', strides=(1, 1), activation='relu', name='conv-2')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool-2')(x)
# print(x.shape)

x=Dropout(0.2, name = 'dropout-1')(x)
x=Flatten(name='flatten')(x)
x=Dense(4, activation='softmax', name='recognition')(x)
model=Model(inp,x)

model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['accuracy']) 
model.summary() 

modelfilepath='model_wzw.hdf5'

earlyStop=EarlyStopping(monitor='val_accuracy',
                        patience=30,
                        verbose=1, 
                        mode='auto')
checkpoint = ModelCheckpoint(modelfilepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                            mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1, 
                              patience=30, 
                              verbose=1, 
                              mode='auto', 
                              min_delta=0.00001, 
                              cooldown=0, 
                              min_lr=0)
history=model.fit(trainX,trainY,validation_split=0.1,epochs=100,batch_size=32,callbacks=[checkpoint,reduce_lr,earlyStop])


# save_model(modelfilepath, r'D:/手势控制系统搭建/机器学习算法模型/CNN50帧模型')



# print('train')
testmodel=load_model(modelfilepath)
#iterate through the test set

#*****混淆矩阵输出*****#
from sklearn.metrics import confusion_matrix
def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(list(y_true), list(y_pred), labels = labels)
    print ("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print ("labels"," ",end='')
    for i in range(len(labels)):
        print (labels[i]," ",end='')
    print('\n')
    for i in range(len(conf_mat)):
        print (i," ",end='')
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j]," ",end='')
        print('\n')
    print

y_pred = []
errorCount = 0.0
for i in range(x_test.shape[0]):
    


    testX=x_test.reshape(-1,50,62,1)
    # print(testmodel.predict(testX))
    
    testX1 = []
    testX1.append(testX[i])
    testX1=np.array(testX1)
    # print(classifierResult)
    classNumStr=y_test[i]
    classifierResult = np.argmax(testmodel.predict(testX1))
    y_pred.append(classifierResult)
    print("CNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, classNumStr))
    if (classifierResult != classNumStr): errorCount += 1.0    
    

print("\n辨识错误数量为: %d" % errorCount)
print("\n辨识率为: %f ％" % ((1 - errorCount / float(x_test.shape[0])) * 100))

my_confusion_matrix( y_test, y_pred)

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
