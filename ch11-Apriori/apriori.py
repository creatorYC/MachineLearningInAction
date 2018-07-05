# coding: utf-8

from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 构建集合C1, C1 是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))   # 不可改变的集合，可作为字典的键


# 数据集、候选集、最小支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                num = ssCnt.get(can, 0)
                ssCnt[can] = num + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# 生成包含k个元素的候选项集列表
def aprioriGen(Lk, k):  # 频繁项集列表、项集元素个数
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:    # 前k-2项相同则合并集合
                retList.append(Lk[i] | Lk[j])
    return retList


# 生成候选项集列表
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


if __name__ == '__main__':
    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # D = list(map(set, dataSet))
    # L1, supportData = scanD(D, C1, 0.5)
    # print(L1)
    L, supportData = apriori(dataSet)
    print(L)