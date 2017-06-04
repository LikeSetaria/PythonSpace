# -*- coding: utf8 -*-
from numpy import *
#读取文本数据
def loadDataSet():
    dataMat=[]; labelMat=[];
    fr=open('D:\\Whu\\Python\\machinelearninginaction\\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append(1.0,float(lineArr[0]),float(lineArr[1]))
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
#定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))
#梯度上升算法
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    lableMat=mat(classLabels)
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(lableMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
#画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()


