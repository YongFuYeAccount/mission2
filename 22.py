# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import torch
import numpy as np
label=[[1,0,0],[1,,0],[1,0,0]]
label = np.array(label)#<class 'numpy.ndarray'>
print(type(label))
print(label)
label_tensor=torch.LongTensor(label)
print(label_tensor)
print(label_tensor.dtype)