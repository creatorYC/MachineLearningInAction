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


# ------------------------ 预测鲍鱼年龄 ---------------------------
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


# 岭回归 计算回归系数
def ridgeRegression(xMat, yMat, lambd=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lambd
    if linalg.det(denom) == 0.0:
        print("the matrix can not do inverse.")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


# 在一组lambda上测试结果
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)   # mean: 求平均值 0：压缩行，求列平均值
    # 数据标准化处理
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    # var(X,W): W可以取0或1：取0求样本方差的无偏估计值(除以N-1)，对应取1求得的是方差(除以N)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegression(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):   # regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)   # calc mean then subtract it off
    inVar = var(inMat, 0)      # calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


# 前项逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIter=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIter, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIter):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat



if __name__ == '__main__':
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # print(ws)
    # showPlot(xArr, yArr)

    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # xMat = mat(xArr)
    # strInd = xMat[:, 1].argsort(0)
    # xSort = xMat[strInd][:, 0, :]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(xSort[:, 1], yHat[strInd])
    # ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    # plt.show()

    abX, abY = loadDataSet('abalone.txt')
    # ridgeWeights = ridgeTest(abX, abY)
    # ax.plot(ridgeWeights)
    # plt.show()

    stageWise(abX, abY, 0.01, 200)




    