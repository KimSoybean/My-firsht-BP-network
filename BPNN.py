# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 19:36:42 2017

@author: zhurui
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:47:03 2017

@author: zhurui
"""

# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.io as sio
import time
f1=open('D:\\zhurui\\NN\\0.1_0.1_method1.txt','w')
cccnt=0
for i in range(1000):
    cccnt=cccnt+1
    print(cccnt)
    # 读入数据
    ################################################################################################
    #print("输入样本文件名（需放在程序目录下）")
    filename = 'mnist_train.mat'     # raw_input() # 换成raw_input()可自由输入文件名
    sample = sio.loadmat(filename)
    sample = sample["mnist_train"]
    sample /= 256.0       # 特征向量归一化
    
    #print("输入标签文件名（需放在程序目录下）")
    filename = 'mnist_train_labels.mat'   # raw_input() # 换成raw_input()可自由输入文件名
    label = sio.loadmat(filename)
    label = label["mnist_train_labels"]
    t_right = np.zeros(10)
    t_numbers = np.zeros(10)
    ##################################################################################################
    
    
    # 神经网络配置
    ##################################################################################################
    samp_num = len(sample)      # 样本总数
    inp_num = len(sample[0])    # 输入层节点数
    out_num = 10                # 输出节点数
    hid_num = 6  # 隐层节点数(经验公式)
    w1 = 0.2*np.random.random((inp_num, hid_num))- 0.1   # 初始化输入层权矩阵
    #w1=np.ones((inp_num,hid_num))*(0.2)/inp_num
    #w1=np.random.normal(size=(inp_num, hid_num))*0.2
    #w1=np.random.randn(inp_num, hid_num)*0.2
                
    w2 = 0.2*np.random.random((hid_num, out_num))- 0.1   # 初始化隐层权矩阵
    #w2=np.ones((hid_num,out_num))*(0.2)/hid_num
    #w2=np.random.normal(size=(hid_num, out_num))*0.2
    #w2=np.random.randn(hid_num, out_num)*0.2    
    
    hid_offset = np.zeros(hid_num)     # 隐层偏置向量
    out_offset = np.zeros(out_num)     # 输出层偏置向量
    inp_lrate = 0.1             # 输入层权值学习率
    hid_lrate = 0.1             # 隐层学权值习率
    err_th = 0.9                # 学习误差门限
    
    #print(w1)
    #print(w2)
    ###################################################################################################
    
    # 必要函数定义
    ###################################################################################################
    def get_act(x):
        act_vec = []
        for i in x:
            act_vec.append(1/(1+math.exp(-i)))
        act_vec = np.array(act_vec)
        return act_vec
    
    def get_err(e):
        return 0.5*np.dot(e,e)
    
    
    ###################################################################################################
    
    # 训练——可使用err_th与get_err() 配合，提前结束训练过程
    ###################################################################################################
    start=time.time()
    ccnt=0
    while 1:    
        ccnt=ccnt+1
        t_right = np.zeros(10)
        t_numbers = np.zeros(10)
        for count in range(0, samp_num):
            #print(count)
            t_label = np.zeros(out_num)
            t_label[label[count]] = 1
            #前向过程
            hid_value = np.dot(sample[count], w1) + hid_offset       # 隐层值  784*1 784*6    +6*1
            hid_act = get_act(hid_value)                # 隐层激活值
            out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
            out_act = get_act(out_value)                # 输出层激活值
        
            #后向过程
            e = t_label - out_act                          # 输出值与真值间的误差
            out_delta = e *out_act *(1-out_act )                                    # 输出层delta计算
            hid_delta = (hid_act)*(1-hid_act) * np.dot(w2, out_delta)                   # 隐层delta计算
            for i in range(0, out_num):
                w2[:,i] += hid_lrate * out_delta[i] * hid_act   # 更新隐层到输出层权向量
            for i in range(0, hid_num):
                w1[:,i] += inp_lrate * hid_delta[i] * sample[count]      # 更新输出层到隐层的权向量
            
            out_offset += hid_lrate * out_delta                             # 输出层偏置更新
            hid_offset += inp_lrate * hid_delta
            
            if np.argmax(out_act) == label[count]:
                t_right[label[count]] += 1
    
        for i in label:
            t_numbers[i] += 1
                     
        #print(t_right)
        #print(t_numbers)
        t_result = t_right/t_numbers
        sum = t_right.sum()
        #print(t_result)
        t_err=sum/len(sample)
        #print(t_err)
        #print(ccnt)
        if (t_err>err_th):     break
        if (ccnt>500 ):        break
    finish=time.time()
    train_time=finish-start
    #print(train_time)
    ###################################################################################################
    
    # 测试网络
    ###################################################################################################
    filename = 'mnist_test.mat'  # raw_input() # 换成raw_input()可自由输入文件名
    test = sio.loadmat(filename)
    test_s = test["mnist_test"]
    test_s /= 256.0
    
    filename = 'mnist_test_labels.mat'  # raw_input() # 换成raw_input()可自由输入文件名
    testlabel = sio.loadmat(filename)
    test_l = testlabel["mnist_test_labels"]
    right = np.zeros(10)
    numbers = np.zeros(10)
                                        # 以上读入测试数据
    # 统计测试数据中各个数字的数目
    for i in test_l:
        numbers[i] += 1
    
    for count in range(len(test_s)):
        hid_value = np.dot(test_s[count], w1) + hid_offset       # 隐层值
        hid_act = get_act(hid_value)                # 隐层激活值
        out_value = np.dot(hid_act, w2) + out_offset             # 输出层值
        out_act = get_act(out_value)                # 输出层激活值
        if np.argmax(out_act) == test_l[count]:
            right[test_l[count]] += 1
    #print(right)
    #print(numbers)
    result = right/numbers
    sum = right.sum()
    #print(result)
    test_err=sum/len(test_s)
    #print(test_err)
    train_time=str(train_time)
    t_err=str(t_err)
    ccnt=str(ccnt)
    test_err=str(test_err)
    temp=[]
    temp.append(train_time)
    temp[0]=temp[0]+' '
    temp.append(t_err)
    temp[1]=temp[1]+' '
    temp.append(ccnt)
    temp[2]=temp[2]+' '
    temp.append(test_err)
    temp[3]=temp[3]+'\n'
    for i in range(4):	
        f1.write(temp[i])

f1.close()
    