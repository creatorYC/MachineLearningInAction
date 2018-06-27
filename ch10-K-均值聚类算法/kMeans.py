# coding: utf-8


from numpy import *


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 计算向量的距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))


# 随机构建质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# K-均值聚类算法
def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    # 簇分配结果矩阵，一列存储簇索引值，一列存储误差(当前点到簇质心的距离)
    clusterAssment = mat(zeros((m, 2)))     
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 寻找最近的质心
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        # 更新质心位置
        for cent in range(k):
            # 属于当前簇的所有点
            ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]
            centroids[cent, :] = mean(ptsInCluster, axis=0) # 沿列方向计算均值
    return centroids, clusterAssment


if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet.txt'))
    # randCent(dataMat, 2)
    # print(distEclud(dataMat[0], dataMat[1]))
    centriods, clusterAssment = kMeans(dataMat, 4)
    # print(centriods)
    # print(clusterAssment)