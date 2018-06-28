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
        # print(centroids)
        # 更新质心位置
        for cent in range(k):
            # 属于当前簇的所有点
            ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]
            centroids[cent, :] = mean(ptsInCluster, axis=0) # 沿列方向计算均值
    return centroids, clusterAssment


# 二分K-均值聚类算法
def biKmeans(dataSet, k, distMeans=distEclud):
    m = shape(dataSet)[0]
    # 簇分配结果矩阵，一列存储簇索引值，一列存储平方误差
    clusterAssment = mat(zeros((m, 2)))
    # 创建初始簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]   # 整个数据集的质心
    cenList = [centroid0]   # 保存所有质心
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :]) ** 2
    while (len(cenList) < k):
        lowestSSE = inf
        for i in range(len(cenList)):   # 划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A==i)[0], :]
            centroidMat, splitClusterAssment = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClusterAssment[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A!=i)[0], 1])
            print("sseSplit and sseNoSplit: ", sseSplit, sseNotSplit)
            # 划分后两个簇的误差与剩余数据集的误差之后作为本次划分的误差
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSlit = i
                bestNewCents = centroidMat
                bestClusAssment = splitClusterAssment.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将新划分得到的编号为0和1的两个簇的编号修改为划分簇的编号和新加簇的编号
        bestClusAssment[nonzero(bestClusAssment[:, 0].A==1)[0], 0] = len(cenList)
        bestClusAssment[nonzero(bestClusAssment[:, 0].A==0)[0], 0] = bestCentToSlit
        print("the bestCenToSplit is: ", bestCentToSlit)
        print("the len of bestClusAssement is: ", len(bestClusAssment))
        # 更新簇的分配结果
        cenList[bestCentToSlit] = bestNewCents[0, :]
        cenList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A==bestCentToSlit)[0], :] = bestClusAssment
    return cenList, clusterAssment


if __name__ == '__main__':
    # dataMat = mat(loadDataSet('testSet.txt'))
    # randCent(dataMat, 2)
    # print(distEclud(dataMat[0], dataMat[1]))
    # centriods, clusterAssment = kMeans(dataMat, 4)

    dataMat2 = mat(loadDataSet('testSet2.txt'))
    centList, newAssment = biKmeans(dataMat2, 3)
    print(centList)
