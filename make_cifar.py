# -*- coding: utf-8 -*-
"""
Created on 20240402 
创建类似CIFAR10的训练数据集
@author: liwei
"""
import numpy as np
from PIL import Image
from os import listdir
import os
import  pickle
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# 初始化相关路径和文件名
flag='eval'  # 指定数据集的类型，如训练集或测试集,train或eval
folderOriginal="data\\original_{}".format(flag)  # 原始图片数据集的文件夹路径
folder32x32="data\\picture32x32"  # 尺寸为32x32的图片数据集文件夹路径
binPath="data\\target"  # 目标二进制文件路径

def getLabel(fname):
    """
    参数:   fname: 文件名。
    返回值: 
    label: 从文件名中提取的标签值。
    例:    getLabel("1_0.jpg")返回1
    """
    fileStr=fname.split(".")[0]
    label=int(fileStr.split("_")[0])
    return label

def img_transform(foldPath,imgList,labels,classes):
    itemsInFolder = listdir(foldPath)
    num=len(itemsInFolder)
    for i in range (0,num):
        itemName = itemsInFolder[i]
        itemPath = "{}\\{}".format(foldPath,itemName)
        if os.path.isdir(itemPath):
            img_transform(itemPath,imgList,labels,classes)    
        elif os.path.isfile(itemPath) and itemName.endswith(".jpg") :
            label=getLabel(itemName)
            im=Image.open(itemPath)
            x=32
            y=32
            out = im.resize((x,y),Image.LANCZOS)
            relativePath = "{}\\{}".format(flag,label)
            savePath = "{}\\{}".format(folder32x32,relativePath)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            saveFullPath = "{}\\{}".format(savePath,itemName)
            if os.path.exists(saveFullPath):
                os.remove(saveFullPath)
            out.save(saveFullPath)
            imgList.append( "{}\\{}".format(relativePath,itemName)) 
            labels.append(label)
            if label not in classes:
                classes.append(label)

def makeMyCf(imgList,labels,classes):
    data={}
    imgs=[]
    listFileName=[]
    num=len(imgList)
    for k in range(0,num):
        im=Image.open("{}\\{}".format(folder32x32,imgList[k]))
        imgs.append(im)
        print("image"+str(k+1)+"saved.")
        listFileName.append(imgList[k].encode('utf-8'))
        
    data.setdefault('labels'.encode('utf-8'),labels)
    data.setdefault('imgs'.encode('utf-8'),imgs)
    data.setdefault('filenames'.encode('utf-8'),listFileName)
    data.setdefault('classes'.encode('utf-8'),classes)
    
    output = open("{}\{}.bin".format(binPath,flag), 'wb')
    pickle.dump(data, output)
    output.close()

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, L in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':

    imgList=[]
    labels=[]
    classes=[]

    img_transform(folderOriginal,imgList,labels,classes)
    makeMyCf(imgList,labels,classes)

    # 求所有样板的均值和方差
    # train_dataset = ImageFolder(root="{}\\{}".format(folder32x32,"train"), transform= transforms.ToTensor())
    # print(getStat(train_dataset))
    
