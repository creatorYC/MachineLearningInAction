# coding: utf-8
from math import log
import operator

# 计算给定数据集的香农熵
def calcShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 数据的最后一列是类别标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1   # 特征个数
    baseEntropy = calcShannonEntropy(dataSet)   # 原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for val in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEntropy(subDataSet)
        # 计算最好的信息增益
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最好特征划分的索引值


# 多数表决方法决定叶子节点的分类(适用于数据集已经处理了所有属性，但类标签依然不唯一的情况)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止划分，返回此唯一类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)  # 本次划分最好特征的索引
    bestFeatureLabel = labels[bestFeature]      # 本次划分最好特征的值
    # 使用字典存储决策树的信息
    myTree = {bestFeatureLabel : {}}
    del(labels[bestFeature])
    # 得到列表包含的所有属性值
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featureValues)
    # 划分每个分支
    for value in uniqueVals:
        subLabels = labels[:]
        nextDataSet = splitDataSet(dataSet, bestFeature, value)
        myTree[bestFeatureLabel][value] = createTree(nextDataSet, subLabels)
    return myTree