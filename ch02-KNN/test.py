# coding: utf-8
import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
# group, labels = kNN.createDataSet()
# print(kNN.classify0([1, 1], group, labels, 3))
# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # 散点图(x,y,点的大小，点的颜色)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#             15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.xlabel(u'每年获取的飞行常客里程数')
# plt.ylabel(u'玩视频游戏所耗时间百分比')
# plt.show()
# kNN.classifyPerson()
# testVector = kNN.img2vector('testDigits/0_13.txt')
# print(testVector[0, 0:31])
kNN.handwritingClassTest()