import Data.LoadData as loadData
import KNN.KNN as KNN
import SVM.SVM as SVM
import numpy as np


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
        for i in range(400):
            index = np.random.randint(0, len(testOX))
            self.testX.append(testOX[index])
            self.testY.append(testOY[index])


if __name__ == '__main__':
    cf = Classifier()
    # KNeighborsClassifier = KNN.knnPredict(cf.trainXSet, cf.trainYSet, cf.validXSet, cf.validYSet,
    #                                       cf.testX, cf.testY, cf.trainNum, cf.validNum, cf.batch_size)
    # print("KNeighborsClassifier: {}".format(KNeighborsClassifier))
    SVM = SVM.SVMPredict(cf.trainXSet, cf.trainYSet, cf.testX, cf.testY)
