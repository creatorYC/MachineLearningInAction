# coding: utf-8

import bayes

listPosts, listClasses = bayes.loadDataSet()
vocabList = bayes.createVocabList(listPosts)
# print(vocabList)
# print(bayes.setOfWords2Vec(vocabList, listPosts[0]))
trainMat = []
for doc in listPosts:
    trainMat.append(bayes.setOfWords2Vec(vocabList, doc))

p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
print(pAb)
print(p0V)