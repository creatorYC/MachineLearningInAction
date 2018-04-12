# coding: utf-8
import trees

myData, labels = trees.createDataSet()
# print(trees.calcShannonEntropy(myData))
# print(trees.splitDataSet(myData, 0, 1))
# print(trees.chooseBestFeatureToSplit(myData))
myTree = trees.createTree(myData, labels)
print(myTree)