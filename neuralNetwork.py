# Neural Network

import numpy as np

numSteps = 10000
hSize = 10
#stepSize = 1e-5

def main():
    xTrain, yTrain = loadTxtFile('netflix-train.txt')
#    hThetas, yThetas = train(xTrain, yTrain)
    xTest, yTest = loadTxtFile('netflix-test.txt')
#    test(hThetas, yThetas, xTest, yTest)

    stepSize = 0.01
    for i in range(30):
        print '=' * 20, 'Step size =', stepSize, '=' * 20
        hThetas, yThetas = train(xTrain, yTrain, stepSize)
        print ''
        test(hThetas, yThetas, xTest, yTest)
        print ''
        stepSize *= 0.1

# x is a m x n matrix where m is number of samples and n is number of input features
# y is a m x 1 matrix where m is number of samples
# returns hThetas, a n x z matrix where n is number of input features and z is the number of hidden features
def train(x, y, stepSize):
    x = np.array(x)
    y = np.array(y)
    
    numSamples = len(x)
    numFeatures = len(x[0])
    
    # Xavier Initialization
    hStd = (2.0 / (numFeatures + hSize)) ** 0.5
    hThetas = np.random.normal(0, hStd, (numFeatures, hSize))
    yStd = (2.0 / (hSize + 1)) ** 0.5
    yThetas = np.random.normal(0, yStd, (hSize, 1))
    
    for j in range(numSteps + 1):
        h = sigmoid(np.matmul(x, hThetas))
        yHat = sigmoid(np.matmul(h, yThetas))
        hGradients = np.transpose(yThetas) * np.matmul(np.transpose(x), (y - yHat) * (h * (1 - h)))
        yGradients = np.matmul(np.transpose(h), y - yHat)
        
        hThetas += stepSize * hGradients
        yThetas += stepSize * yGradients
        
        if j % 500 == 0:
            print 'Iteration {} -> Log likelihood of data: {}'.format(j, LL(y, yHat))
    
    return hThetas, yThetas

# computes sigmoid over all values of x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# returns the LL
# y and yHat are both m x 1 column vectors where m is number of samples
# getting an error with np.dot(y, np.log(yHat)) because it is performing
# matrix multiplication with (m x 1) * (m x 1) -> use transpose instead
def LL(y, yHat):
    return np.dot(np.transpose(y), np.log(yHat)) + np.dot(np.transpose(1 - y), np.log(1 - yHat))

# loads a .txt file
# returns m x n matrix of x's where m is number of samples and n is number of input features
# and an m x 1 vector of y's where m is number of samples
def loadTxtFile(filename):
    numFeatures = 0
    numSamples = 0
    x = []
    y = []
    with open(filename, 'r') as f:
        numFeatures = int(f.readline())
        numSamples = int(f.readline())
        for line in f:
            sample = np.array(map(lambda x: int(x[0]), line.split()))
            x.append(np.insert(sample[:-1], 0, 1))
            y.append([sample[-1]])
    return x, y

def test(hThetas, yThetas, x, y):
    tested0 = 0
    correct0 = 0
    tested1 = 0
    correct1 = 0
    for i in range(len(x)):
        sample = x[i]
        answer = y[i][0]
        prediction = predict(hThetas, yThetas, sample)
        if answer == 0:
            tested0 += 1
            if prediction == 0:
                correct0 += 1
        if answer == 1:
            tested1 += 1
            if prediction == 1:
                correct1 += 1
    tested = tested0 + tested1
    correct = correct0 + correct1
    print 'Class 0: tested {}, correctly classified {}'.format(tested0, correct0)
    print 'Class 1: tested {}, correctly classified {}'.format(tested1, correct1)
    print 'Overall: tested {}, correctly classified {}'.format(tested, correct)
    print 'Accuracy =', float(correct) / float(tested)

# picks value of y that maximizes the likelihood
# of the sample having its values
def predict(hThetas, yThetas, sample):
    h = sigmoid(np.matmul(sample, hThetas))
    yHat = sigmoid(np.matmul(h, yThetas))
    return yHat[0] > 0.5

if __name__ == '__main__':
    main()
