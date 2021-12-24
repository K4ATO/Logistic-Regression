import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from pandas.plotting import scatter_matrix

csvFile = "H:\Hitler\FCAI\\ML\Assignment1\heart.csv"
data = pd.read_csv(csvFile, usecols=['trestbps', 'chol', 'thalach', 'oldpeak', 'target'])


def prepdata(data):
    # separate X (training data) from y (target variable)
    Xtrain = data.loc[:, ['trestbps', 'chol', 'thalach', 'oldpeak']]
    Ytrain = data.loc[:, ['target']]

    # normalizing the data
    Xtrain = (Xtrain - Xtrain.mean()) / Xtrain.std()

    # add ones column
    Xtrain.insert(0, 'Ones', 1)

    # convert to matrices
    Xtrain = np.matrix(Xtrain.values)
    Ytrain = np.matrix(Ytrain.values)

    Xtest = data.loc[(len(data) * 80 / 100):, ['trestbps', 'chol', 'thalach', 'oldpeak']]
    Ytest = data.loc[(len(data) * 80 / 100):, ['target']]

    # normalizing the data
    Xtest = (Xtest - Xtest.mean()) / Xtest.std()

    # add ones column
    Xtest.insert(0, 'Ones', 1)

    # convert to matrices
    Xtest = np.matrix(Xtest.values)
    Ytest = np.matrix(Ytest.values)

    # initialize theta
    initial_theta = np.zeros((Xtrain.shape[1], 1))

    return Xtrain, Ytrain, Xtest, Ytest, initial_theta


# Hypothesis Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Cost Function
def cost(theta, X, y, learningRate):
    first = np.multiply(y, np.log(sigmoid(X * theta)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
    #reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(-first - second)


# a)
# Gradient Descent
def gradientdescent(X, Y, theta, alpha, iterations):
    j_history = []
    m = len(X)
    theta_temp = np.zeros((X.shape[1], 1))
    for j in range(iterations):
        for i in range(X.shape[1]):
            theta_temp[i] = theta[i] - (alpha * sum((sigmoid(np.dot(X, theta)) - Y).T * (X[:, i] * (1 / m))))
        theta = theta_temp.copy()
        j_history.append(cost(theta, X, Y, alpha))
    return theta, j_history


def printcost(history):
    print(history)


def predict(X_test, theta):
    h = sigmoid(theta.T * X_test.T) >= 0.5
    h = h.T
    return h


def printpredictions(h, Y_test):
    Y_test = Y_test >= 0.5
    print("actual value: ", Y_test)
    print("predicted: ", h)


# initialize variables for learning rate and iterations
iterations = 3000
alpha = 0.1

# b)
# preparing the data
X_train, Y_train, X_test, Y_test, initial_theta = prepdata(data)

# perform gradient descent to "fit" the model parameters
theta, history = gradientdescent(X_train, Y_train, initial_theta, alpha, iterations)

# printing the error in every iteration
printcost(history)

# c)
h = predict(X_test, theta)
printpredictions(h, Y_test)

# d)
alpha = 0.003
theta1, history1 = gradientdescent(X_train, Y_train, initial_theta, alpha, iterations)

alpha = 0.1
theta2, history2 = gradientdescent(X_train, Y_train, initial_theta, alpha, iterations)

alpha = 0.03
theta3, history3 = gradientdescent(X_train, Y_train, initial_theta, alpha, iterations)

alpha = 0.001
theta4, history4 = gradientdescent(X_train, Y_train, initial_theta, alpha, iterations)

plt.plot(range(1, iterations +1), history1, color ='purple', label = 'alpha = 0.003')
plt.plot(range(1, iterations +1), history2, color ='red', label = 'alpha = 0.1')
plt.plot(range(1, iterations +1), history3, color ='green', label = 'alpha = 0.03')
plt.plot(range(1, iterations +1), history4, color ='yellow', label = 'alpha = 0.001')

plt.rcParams["figure.figsize"] = (5,5)
plt.grid()
plt.xlabel("Number of iterations")
plt.ylabel("cost (J)")
plt.title("Effect of Learning Rate On Convergence of Gradient Descent")
plt.legend()