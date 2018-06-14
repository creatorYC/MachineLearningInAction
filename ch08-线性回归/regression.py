# coding: utf-8

from numpy import *


# 数据导入
def loadDataSet(filename):
    # 特征数
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 计算最佳拟合直线的回归系数
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # 矩阵对应的行列式
        print("the matrix can not do inverse.")
        return
    ws = xTx.I * (xMat.T * yMat)
    # ws = linalg.solve(xTx, xMat.T*yMat)   # solve: 解方程
    return ws


# 绘制最佳拟合直线图
def showPlot(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 散点    # A: 矩阵转化为对应的数组
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws   # y的预测值
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


# 局部加权线性回归函数
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))     # eye: 单位矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))  # 高斯核
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("the matrix can not do inverse.")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    # print(ws)
    # showPlot(xArr, yArr)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    strInd = xMat[:, 1].argsort(0)
    xSort = xMat[strInd][:, 0, :]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[strInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()

    