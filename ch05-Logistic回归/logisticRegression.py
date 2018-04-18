# coding: utf-8

from numpy import *
import matplotlib.pyplot as plt


# 读取文件内容并格式化数据
def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt') as f:
        for line in f.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat


# 阶跃函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 使用梯度上升算法计算最佳回归系数
def gradientAscent(dataMatIn, classLabels):
    # 转换为Numpy矩阵类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T   # 矩阵转置
    m, n = shape(dataMatrix)
    alpha = 0.001   # 向目标移动的步长
    numCycles = 500 # 迭代次数
    weights = ones((n, 1))
    # 计算真实类别与预测类别的差值, 按照该差值的方向调整回归系数
    for k in range(numCycles):
        h = sigmoid(dataMatrix * weights)   # 矩阵相乘
        error = (labelMat - h)  # 向量相减
        weights = weights + alpha * dataMatrix.T * error
    return weights

# 使用随机梯度上升算法计算最佳回归系数
# 一次只用一个样本点来更新回归系数, 能对数据进行增量更新, 是一个"在线学习"算法
# 但因为数据集可能不是线性可分, 在迭代的时候可能导致回归系数抖动, 收敛速度慢
def stocGradientAscent_0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 使用改进的随机梯度上升算法计算最佳回归系数
# 步长alpha每次都会调整
# 通过随机选取样本来更新回归系数, 减少周期型抖动, 增加收敛速度
def stocGradientAscent_1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 步长每次迭代都会减少 1/(j+i)
            # j 为迭代次数, i 为样本点的下标
            alpha = 4/(1.0+j+i)+0.01  # 常数使得 alpha 永远不会减少到 0
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# 绘制最佳拟合直线
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    m, _ = shape(dataArr)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(m):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制散点
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 绘制直线
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = gradientAscent(dataArr, labelMat)
    # plotBestFit(weights.getA()) # getA(): 矩阵变量转化为ndarray类型的变量
    weights = stocGradientAscent_1(array(dataArr), labelMat)
    plotBestFit(weights)















