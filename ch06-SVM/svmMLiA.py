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


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    # print(labelArr)
    b, alphas = simpleSMO(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas[alphas>0])