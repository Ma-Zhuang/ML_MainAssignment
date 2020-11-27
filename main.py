import Data.LoadData as loadData
import KNN.KNN as KNN
import SVM.SVM as SVM
import DecisionTree.DecisionTree as DecisionTree
import numpy as np
import matplotlib.pyplot as plt
import time


class Classifier:
    def __init__(self):
        self.trainNum = 48000
        self.validNum = 12000
        trainX, trainY, testOX, testOY = loadData.read_data()
        self.trainXSet = trainX[:self.trainNum]
        self.trainYSet = trainY[:self.trainNum]
        self.validXSet = trainX[self.trainNum:]
        self.validYSet = trainY[self.trainNum:]
        self.batch_size = 120
        self.testX = []
        self.testY = []
        for i in range(1, 11):
            index = i * 1000
            rand = np.random.randint(index - 1000, index, 500)
            self.testX.append(testOX[rand])
            self.testY.append(testOY[rand])


if __name__ == '__main__':
    treeAcc = []
    svmAcc = []
    count = []
    cf = Classifier()
    treeTime = time.time()
    depth = DecisionTree.treeValid(cf.trainXSet, cf.trainYSet, cf.validXSet, cf.validYSet, cf.trainNum, cf.validNum,
                                   cf.batch_size)
    treeModel = DecisionTree.fit(cf.trainXSet, cf.trainYSet, depth)
    for i in range(len(cf.testX)):
        tacc = DecisionTree.treePredict(treeModel, cf.testX[i], cf.testY[i])
        count.append(i)
        treeAcc.append(tacc)
    print("tree time:{}, depth:{}".format((time.time() - treeTime), depth))
    print("\nDecisionTree辨识率为: %f ％" % np.mean(treeAcc))

    svmTime = time.time()
    gama = SVM.svmValid(cf.trainXSet, cf.trainYSet, cf.validXSet, cf.validYSet, cf.trainNum, cf.validNum,
                        cf.batch_size)
    print(gama)
    svmModel = SVM.fit(cf.trainXSet, cf.trainYSet, gama)
    for i in range(len(cf.testX)):
        sacc = SVM.SVMPredict(svmModel, cf.testX[i], cf.testY[i])
        svmAcc.append(sacc)
    print("svm time:{}, gama:{}".format((time.time() - svmTime), gama))
    print("\nSVM辨识率为: %f ％" % np.mean(svmAcc))
    knnTime = time.time()
    Knnacc = KNN.knnPredict(cf.trainXSet, cf.trainYSet, cf.validXSet, cf.validYSet,
                            cf.testX, cf.testY, cf.trainNum, cf.validNum, cf.batch_size)
    print("knn time:{}".format(time.time() - knnTime))
    print("\nKNN辨识率为: %f ％" % np.mean(Knnacc))
    plt.plot(count, treeAcc, color='r', label='Decision Tree accuracy')
    plt.plot(count, svmAcc, color='g', label='SVM accuracy')
    plt.plot(count, Knnacc, color='b', label='KNN accuracy')
    plt.legend()
    plt.title('Correct rate', fontsize=24)
    plt.xlabel('Train Count', fontsize=14)
    plt.ylabel('Correct rate(%)', fontsize=14)
    plt.show()
