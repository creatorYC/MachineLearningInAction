# coding: utf-8
from numpy import *


def loadSimpleData():
    dataMat = matrix(
        [
            [1., 2.1],
            [2., 1.1],
            [1.3, 1.],
            [1., 1.],
            [2., 1.]
        ]
    )
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


# 决定将小于或大于阈值的一侧作为-1类
def stumpClassify(dataMatrix, dimen, threshVal, threshInq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshInq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] >= threshVal] = -1.0
    return retArray


# 选取让误差率最低的阈值来设计基本分类器
def buildStump(dataArr, classLabels, D):    # D: 权重向量
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):  # 对每个特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inEqual in ['lt', 'gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictVals = stumpClassify(dataMatrix, i, threshVal, inEqual)
                errArray = mat(ones((m, 1)))
                errArray[predictVals==labelMat] = 0
                weightError = D.T * errArray    # 错误率
                # print("split: dim %d, thresh %.2f, inequ: %s, weightedErr: %.3f" 
                #     % (i, threshVal, inEqual, weightError))
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inEqual
    return bestStump, minError, bestClassEst


# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIter=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))    # 组合各个弱分类器
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D: ", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst: ", classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(   # 错误率累加
            sign(aggClassEst) != mat(classLabels).T, ones((m, 1))
        )
        errorRate = aggErrors.sum() / m
        # print("total Error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr


# 利用训练好的多个弱分类器进行分类
def adaClassify(dataToClass, classifierArray):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArray)):
        classEst = stumpClassify(
            dataMatrix,
            classifierArray[i]['dim'],
            classifierArray[i]['thresh'],
            classifierArray[i]['ineq']
        )
        aggClassEst += classifierArray[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)


if __name__ == '__main__':
    datMat, classLabels = loadSimpleData()
    D = mat(ones((5, 1))/5)
    bestStump, minError, bestClassEst = buildStump(datMat, classLabels, D)
    # print(bestStump)
    # print(bestClassEst)
    # print(minError)
    classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    adaClassify([[5, 5], [0, 0]], classifierArray)
