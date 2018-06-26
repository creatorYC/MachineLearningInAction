# coding: utf-8

from numpy import *


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(list(fltLine))
    return dataMat


# 通过数组过滤方式切分数据集合
def binSplitDataSet(dataSet, featureIndex, value):
    mat0 = dataSet[nonzero(dataSet[:, featureIndex] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, featureIndex] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    return mean(dataSet[:, -1])


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


# 寻找数据的最佳二元切分方式
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]   # 容许的误差下降值
    tolN = ops[1]   # 切分的最少样本数
    # tolist: 矩阵转换成列表,矩阵是一维的时候有tolist()[0]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:     # 如果所有feature值相等则退出
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 如果误差减少不大则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    featureIndex, value = chooseBestSplit(dataSet, leafType, errType, ops)
    if featureIndex == None:
        return value
    retTree = {}
    retTree['spIdx'] = featureIndex
    retTree['spVal'] = value
    lSet, rSet = binSplitDataSet(dataSet, featureIndex, value)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def  getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']) / 2.0


# 回归树剪枝
def prune(tree, testData):
    # 没有数据时对树进行塌陷处理(即返回树平均值)
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spIdx'], tree['spVal'])
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spIdx'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1]-tree['left'], 2)) + \
                        sum(power(rSet[:, -1]-tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 模型树的叶节点生成函数
def linearSolve(dataSet):
    m, n = shape(dataSet)
    # 格式化X，Y中的数据
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError("this matrix can not do inverse, try incresing the second value of ops.")
    ws = xTx.I * (X.T * Y)  # 线性回归
    return ws, X, Y


# 数据不再切分时生成叶节点
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


# 在给定的数据集上计算误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat, 2))


def regTreeEval(model, inData):
    return float(model)


def modelTreeEval(model, inData):
    n = shape(inData)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1] = inData
    return float(X*model)


# 用树回归进行预测
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spIdx']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # myData = loadDataSet('ex0.txt')
    # myMat = mat(myData)
    # retTree = createTree(myMat)
    # print(retTree)
    # myData2 = loadDataSet('exp2.txt')
    # myMat2 = mat(myData2)
    # myTree = createTree(myMat2, modelLeaf, modelErr, ops=(1, 10))
    # print(myTree)

    # ----------- 创建回归树 ---------------
    # corrcoef: 相关系数
    trainMat = mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    # ----------- 创建模型树 ---------------
    myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])