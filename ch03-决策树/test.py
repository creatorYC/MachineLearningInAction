# coding: utf-8
import trees
import treePlotter

myData, labels = trees.createDataSet()
# # print(trees.calcShannonEntropy(myData))
# # print(trees.splitDataSet(myData, 0, 1))
# # print(trees.chooseBestFeatureToSplit(myData))
# myTree = trees.createTree(myData, labels)
# print(myTree)
myTree = treePlotter.retrieveTree(0)
# treePlotter.createPlot(myTree)
print(trees.classify(myTree, labels, [1, 0]))