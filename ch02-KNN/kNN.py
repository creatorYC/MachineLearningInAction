# coding: utf-8
from numpy import *
import operator
from os import listdir

# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 数组的行的维数
    # tile(A, B): 重复A，B次，这里的B可以时int类型也可以是元组类型
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 将inX在行方向上重复dataSetSize次，列方向重复1次
    sqDiffMat = diffMat ** 2
    # axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列 
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    # argsort函数返回的是 数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    sortedClassCount = sorted(classCount.items(), 
                                key=operator.itemgetter(1), # 用于比较的关键字(按第二个元素排序)
                                reverse=True) # 倒序
    return sortedClassCount[0][0]


# --------------------  约会网站数据分析  ----------------------------------
# 将文本记录转换为NumPy的解析程序
def file2matrix(filename):
    fr = open(filename)
    arrLines = fr.readlines()
    numLines = len(arrLines)
    retMat = zeros((numLines, 3))
    classLabelVector = []
    index = 0
    for line in arrLines:
        line = line.strip()
        listFromLine = line.split('\t')
        retMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return retMat, classLabelVector


# 归一化特征值(newVal = (oldVal-minVal)/(maxVal-minVal))
def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 获取列的最小值
    maxVals = dataSet.max(0)    # 获取列的最大值
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 测试分类器效果
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingDataLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  # 前 10% 的数据用于测试
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                        datingDataLabels[numTestVecs:m], 3)
        print(u'分类器返回: %d, 真实结果: %d'
                                        % (classifierResult, datingDataLabels[i]))     
        if(classifierResult != datingDataLabels[i]):
            errorCount += 1.0
    print(u'错误率为: %f' % (errorCount/float(numTestVecs)))    


# 预测函数
def classifyPerson():
    resultList = [u'一点也不喜欢', u'喜欢一点点', u'非常喜欢']
    percentAges = float(input(u"玩视频游戏所耗时间百分比?"))
    ffMiles = float(input(u"每年获取的飞行常客里程数?"))
    iceCream = float(input(u"每年消费的冰淇淋公升数?"))
    datingDataMat, datingDataLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentAges, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingDataLabels, 3)
    print(u"测试结果为", resultList[classifierResult - 1])     


# ------------------------  手写识别系统  ------------------------
# 将图像转为向量
def img2vector(filename):
    retVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            retVector[0, 32*i+j] = int(lineStr[j])
    return retVector


# 手写识别系统测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = classify0(vectorTest, trainMat, hwLabels, 3)
        print(u'分类器返回: %d, 真实结果: %d' % (classifierResult, classNumStr))     
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print(u'错误率为: %f' % (errorCount/float(mTest)))    
