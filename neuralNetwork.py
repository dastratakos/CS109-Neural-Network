# Neural Network

import numpy as np

numSteps = 10000
hSize = 5
#stepSize = 1e-6

def main():
    xTrain, yTrain = loadTxtFile('netflix-train.txt')
#    hThetas, yThetas = train(xTrain, yTrain)
    xTest, yTest = loadTxtFile('netflix-test.txt')
#    test(hThetas, yThetas, xTest, yTest)

    stepSize = 0.01
    for i in range(30):
        print '=' * 20, 'Step size =', stepSize, '=' * 20
        hThetas, yThetas = train(xTrain, yTrain, stepSize)
        test(hThetas, yThetas, xTest, yTest)
        stepSize *= 0.1

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
    
    hOnes = np.ones((hSize, numSamples))
    yOnes = np.ones((1, numSamples))
    
    for j in range(numSteps + 1):
        """
        if j % 500 == 0:
            print '=' * 30 + ' Iteration {} '.format(j) + '=' * 30
            
            print 'Current model parameters:'
            for i in range(len(thetas)):
                print '\\theta_{{{}}} ='.format(i), thetas[i], '\\\\'
            print 'Log likelihood of data: {0:.3f}\n'.format(LL(thetas, x, y))
        """
        h = sigmoid(np.matmul(x, hThetas))
        yHat = sigmoid(np.matmul(h, yThetas))
        hGradients = np.transpose(yThetas) * np.matmul(np.transpose(x), (y - yHat) * (h * (1 - h)))
        yGradients = np.matmul(np.transpose(h), y - yHat)
        
        hThetas += stepSize * hGradients
        yThetas += stepSize * yGradients
    
    return hThetas, yThetas

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def LL(thetas, x, y):
    sum = 0.0
    for i in range(len(x)):
        sum += y[i] * np.log(sigmoid(np.dot(thetas, x[i]))) + (1 - y[i]) * np.log(1 - sigmoid(np.dot(thetas, x[i])))
    return sum

# loads a .txt file
# returns 2D matrix of x's with size n by m
# and a vector of y's with length n
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
    hSize = len(yThetas)
    hOnes = np.ones((hSize, 1))
    
    temp = np.matmul(np.transpose(hThetas), sample)
    temp = np.reshape(temp, (len(temp), 1))
    h = hOnes / (hOnes + (np.exp(-temp)))
    yHat = 1 / (1 + (np.exp(-np.matmul(np.transpose(h), yThetas))))
    
#    return 1 if yHat[0] > 0.5 else 0
    return yHat[0] > 0.5

if __name__ == '__main__':
    main()
