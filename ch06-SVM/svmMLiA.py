# coding: utf-8
from numpy import *

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 返回任一 [0, m) 之间且不等于 i 的数
def randomSelectJ(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 调整大于H或者小于L的alpha值
def adjustAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def estimate(alphas, labelMat, dataMatrix, index, b):
    fx = float(
        multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[index, :].T)
    ) + b
    e = fx - float(labelMat[index])
    return e


# SMO算法简化版(每次循环中选择两个alpha进行优化处理.一旦找到一对合适的alpha,
#  那么就增大其中一个同时减少另外一个)
# toler: 容错率
def simpleSMO(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    b = 0
    m, n = shape(dataMatrix)
    # 初始化alpha向量
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairChanged = 0    # alpha 是否已经优化
        for i in range(m):
            # 计算 alphas[i] 的预测值, 估算其是否可以被优化
            Ei = estimate(alphas, labelMat, dataMatrix, i, b)
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C) or 
                (labelMat[i]*Ei> toler) and (alphas[i]>0)):
                # 选择第二个 alpha[j]
                j = randomSelectJ(i, m)
                # alpha[j] 的预测值
                Ej = estimate(alphas, labelMat, dataMatrix, j, b)
                # 保存旧值以便与调整后比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 调整 alpha[j] 至 (0, C) 之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # 计算 alpha[j] 的最优修改量
                delta = (
                    2.0 * dataMatrix[i, :] * dataMatrix[j, :].T
                    - dataMatrix[i, :] * dataMatrix[i, :].T
                    - dataMatrix[j, :] * dataMatrix[j, :].T
                )
                # 如果 delta==0, 则需要退出for循环的当前迭代过程.
                # 简化版中不处理这种少量出现的特殊情况
                if delta >= 0:
                    print("delta>=0")
                    continue

                # 计算新的 alpha[j]
                alphas[j] -= labelMat[j]*(Ei - Ej) / delta
                alphas[j] = adjustAlpha(alphas[j], H, L)
                # 若 alpha[j] 的改变量太少, 不采用
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough.")
                    continue
                # 对 alpha[i] 做 alpha[j] 同样大小, 方向相反的改变
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                # 给两个 alpha 值设置常量 b
                b1 = (
                    b - Ei
                    - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T
                    - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                )
                b2 = (
                    b - Ej
                    - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T
                    - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                )
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairChanged += 1
                print("iter: %d i: %d, pairsChanged: %d" % (iter, i, alphaPairChanged))
        if alphaPairChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


class optStruct(object):
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toler = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # eCache 第一列是eCache是否有效的标志位，第二列是实际的E值
        self.eCache = mat(zeros((self.m, 2)))


def calcEk(optS, k):
    fXk = float(
        multiply(optS.alphas, optS.labelMat).T
        * (optS.X * optS.X[k, :].T)
    ) + optS.b
    Ek = fXk - float(optS.labelMat[k])
    return Ek


def selectJ(i, optS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    optS.eCache[i] = [1, Ei]    # 设置第i个eCache缓存值
    # nonzeros(a)返回数组a中值不为零的元素的下标,返回值为元组， 
    # 两个值分别为两个维度， 包含了相应维度上非零元素的下标值
    # .A 代表将矩阵转化为array数组类型
    validEcacheList = nonzero(optS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        # 在有效的缓存值中寻找deltaE最大的
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(optS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        # 没有任何有效的eCache缓存值 (如第一轮中)
        j = randomSelectJ(i, optS.m)
        Ej = calcEk(optS, j)
    return j, Ej


def updateEk(optS, k):
    Ek = calcEk(optS, k)
    optS.eCache[k] = [1, Ek]


def innerL(i, optS):
    # 计算 alpha[i] 的预测值, 估算其是否可以被优化
    Ei = calcEk(optS, i)
    if ((optS.labelMat[i]*Ei<-optS.toler) and (optS.alphas[i]<optS.C) or
        (optS.labelMat[i]*Ei>optS.toler) and (optS.alphas[i]>0)):
        j, Ej = selectJ(i, optS, Ei)
        alphaIold = optS.alphas[i].copy()
        alphaJold = optS.alphas[j].copy()
        if (optS.labelMat[i] != optS.labelMat[j]):
            L = max(0, optS.alphas[j]-optS.alphas[i])
            H = min(optS.C, optS.C+optS.alphas[j]-optS.alphas[i])
        else:
            L = max(0, optS.alphas[j]+optS.alphas[i]-optS.C)
            H = min(optS.C, optS.alphas[j]+optS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        delta = (
            2.0 * optS.X[i,:] * optS.X[j,:].T
            - optS.X[i,:] * optS.X[i,:].T
            - optS.X[j,:] * optS.X[j,:].T
        )
        if delta >= 0:
            print("delta >= 0")
            return 0
        optS.alphas[j] -= optS.labelMat[j] * (Ei - Ej) / delta
        optS.alphas[j] = adjustAlpha(optS.alphas[j], H, L)
        updateEk(optS, j)
        if (abs(optS.alphas[j]-alphaJold) < 0.00001):
            print("j not moving enough.")
            return 0
        optS.alphas[i] += optS.labelMat[j]*optS.labelMat[i]*(alphaJold-optS.alphas[j])
        updateEk(optS, i)
        b1 = (
            optS.b - Ei
            - optS.labelMat[i]*(optS.alphas[i]-alphaIold)*optS.X[i,:]*optS.X[i,:].T
            - optS.labelMat[j]*(optS.alphas[j]-alphaJold)*optS.X[i,:]*optS.X[j,:].T
        )
        b2 = (
            optS.b - Ej
            - optS.labelMat[i]*(optS.alphas[i]-alphaIold)*optS.X[i,:]*optS.X[j,:].T
            - optS.labelMat[j]*(optS.alphas[j]-alphaJold)*optS.X[j,:]*optS.X[j,:].T
        )
        if 0 < optS.alphas[i] < optS.C:
            optS.b = b1
        elif 0 < optS.alphas[j] < optS.C:
            optS.b = b2
        else:
            optS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    optS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged>0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历alpha, 使用 innerL 选择 alpha-j, 并在可能是对其进行优化
            for i in range(optS.m):
                alphaPairsChanged += innerL(i, optS)
                print("fullSet, iter: %d i: %d, pairsChanged: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            # 遍历所有非边界(不在边界0或C上)的 alpha
            nonBoundIs = nonzero((optS.alphas.A > 0) * (optS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, optS)
                print("non-bound, iter: %d i: %d, pairsChanged: %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        else:
            print("iteration number: %d" % iter)
    return optS.b, optS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i], X[i,:].T)
    return w


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print(labelArr)
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b)
    # print(alphas[alphas>0])
    ws = calcWs(alphas, dataArr, labelArr)
    # 对数据进行分类
    dataMat = mat(dataArr)
    print(dataMat[0]*mat(ws)+b)
    print(labelArr[0])