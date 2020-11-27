from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np


def treeValid(trainXSet, trainYSet, validXSet, validYSet, trainNum, validNum, batch_size):
    errorCount = 0.0
    acc = []
    k = []
    mTest = np.random.randint(0, validNum, 500)
    for j in range(0, 50):
        k.append(j)  # trainNum // batch_size
        model = DecisionTreeClassifier(max_depth=j + 1)
        model.fit(trainXSet[:2500], trainYSet[:2500])
        for i in range(len(mTest)):
            classifierResult = model.predict(validXSet[mTest[i]].reshape(1, -1))
            # print("KNN得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
            if (classifierResult != validYSet[mTest[i]]): errorCount += 1.0
        acc.append(((1 - errorCount / float(len(mTest))) * 100))
        errorCount = 0.0
        indexTmp = np.argwhere(acc == np.amax(acc))
        index = []
        for i in range(len(indexTmp)):
            index.append(indexTmp[i][0])
    plt.plot(k, acc)
    plt.title('DecisionTree Correct rate', fontsize=24)
    plt.xlabel('Depth', fontsize=14)
    plt.ylabel('Correct rate(%)', fontsize=14)
    plt.show()
    print("\nValid DecisionTree辨识率为: %f ％" % np.mean(acc))
    return int(np.mean(index))


def fit(trainXSet, trainYSet, depth):
    print("start tree")
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(trainXSet, trainYSet)
    print("finished tree")
    return model


def treePredict(model, testX, testY):
    errorCount = 0.0
    mTest = len(testX)
    for i in range(mTest):
        classifierResult = model.predict(testX[i].reshape(1, -1))
        # print("DecisionTree得到的辨识结果是: %d, 实际值是: %d" % (classifierResult, testY[i]))
        if (classifierResult != testY[i]): errorCount += 1.0
    acc = (1 - errorCount / float(mTest)) * 100
    # print("\nDecisionTree辨识错误数量为: %d" % errorCount)
    # print("\nDecisionTree辨识率为: %f ％" % acc)
    return acc
