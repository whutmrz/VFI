# 导入的模块
import torch
import os
import glob
import random
import xml.etree.ElementTree as ET

# 划分数据集

# 数据划分比例
# (训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1

path = 'E:/WHU/VFI-master/Dataset/experiments'

# 按照比例划分数据集
train_per = 0
valid_per = 1

i = 1
for subfolder in os.listdir(path):
    data_list = os.listdir(os.path.join(path, subfolder))
    # random.seed(666)
    # random.shuffle(data_list)
    data_length = len(data_list)

    train_point = int(data_length * train_per)

    # 生成训练集，验证集, 测试集(8 : 2)
    train_list = data_list[:train_point]
    valid_list = data_list[train_point:]

    # 写入文件中
    ftrain = open('E:/WHU/VFI-master/Dataset/experiments/txt/train_{}.txt'.format(i), 'a')
    fvalid = open('E:/WHU/VFI-master/Dataset/experiments/txt/test_{}.txt'.format(i), 'a')


    for i in train_list:
        ftrain.write(i + "\n")
    for j in valid_list:
        fvalid.write(j + "\n")
    i += 1
    ftrain.close()
    fvalid.close()

    print("总数据量:{}, 训练集:{}, 验证集:{}".format(len(data_list), len(train_list), len(valid_list)))
    print("done!")