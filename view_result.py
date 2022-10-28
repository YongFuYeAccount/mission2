# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf

import torch
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

#保存训练的概率为csv
def keepresult(savefilepath, headerlines, real_Labels, pred_class, totalprobality):
    # 保存csv的地址，csv的列名，实际的标签（list），预测的标签（list），各个列别的标签(list)
    csvFile = open(savefilepath, "w",newline="")  # 创建csv文件
    writer = csv.writer(csvFile)  # 创建写的对象
    # 先写入columns_name
    writer.writerow(headerlines)  # 写入列的名称
    for i in range(len(list(real_Labels))):
        probability = list(totalprobality[i])  # 将概率np转化为list
        writer.writerow(
            [real_Labels[i], pred_class[i], probability[0], probability[1], probability[2], probability[3]])
    csvFile.close()

#画混淆矩阵
def draw_cm(y_test,y_pred,labels):
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # 得到混淆矩阵(confusion matrix,简称cm)
    # confusion_matrix 需要的参数：y_true(真实标签),y_pred(预测标签),normalize(归一化,'true', 'pred', 'all')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

#画混淆矩阵
# 相关库
def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        pl.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('semiadda_Frame20_cm')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    plt.show()

# 绘制精确率-召回率曲线
# def plot_precision_recall(recall, precision):
#     plt.step(recall, precision, color='b', alpha=0.2, where='post')
#     plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
#     plt.plot(recall, precision, linewidth=2)
#     plt.xlim([0.0, 1])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('zhaohuilv')
#     plt.ylabel('precision')
#     plt.title('precision-recall carve')
#     plt.show();

#画pr曲线
def drawPre_recall(test_y, y_score):  # test_y为测试数据的类别，y_score是每个类别的概率
    y = test_y
    # 使用label_binarize让数据成为类似多标签的设置
    Y = label_binarize(y, classes=[0, 1, 2, 3])
    n_classes = Y.shape[1]

    # 对每个类别
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])

    # 一个"微观平均": 共同量化所有课程的分数
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(), y_score.ravel())

    average_precision["micro"] = average_precision_score(Y, y_score, average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.01])
    plt.title('single_amp_20Frame: AP={0:0.2f}'.format(average_precision["micro"]))

    # plt.title('Semi-ADDA_20Frame micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    # plt.title(
    #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    # plt.title('pr-curve')
    plt.show()



def plot_result(test_x,test_y,net):
    test_output, _ = net(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    print(pred_y, 'prediction number')
    print(test_y.numpy(), 'real number')
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    print('选取30%init数据作为测试集（验证集数据）',accuracy)

    ##draw cm
    classes = ["bedroom_empty", "bedroom_sit", "toilet_empty", "toilet_sit"]  # classes = list(reversed(classes))print(a,list(reversed(a)))
    label = [0, 1, 2, 3]

    draw_cm(list(pred_y), list(test_y), classes)  # 调用画混淆矩阵函数
    plot_matrix(list(test_y), list(pred_y), label, title=None, thresh=0.8, axis_labels=None)  # 画混淆矩阵方法二

    # 得到分类过程的分类概率
    probability = torch.nn.functional.softmax(test_output, dim=1)  # 计算softmax，即该图片属于各类的概率
    # max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别（输出是tensor）

    # 保存数据需要的参数
    totalprobality = np.round(probability.detach().numpy(), 3)  # 保留概率的三位小数
    # #画pr曲线
    drawPre_recall(test_y, totalprobality)

    # 打印各个类别的结果
    import collections
    data_count = collections.Counter(pred_y)
    print(data_count)









