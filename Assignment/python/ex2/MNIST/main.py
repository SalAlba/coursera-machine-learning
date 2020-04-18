import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import idx2numpy

# from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from sal_timer import timer


class Model:
    def __init__(self, xdim=3, decision=0.5, t0=5, t1=50, alpha=.5, gamma=.9, lambda_=.1, epochs=10, verbose=True):
        self.cost_history = np.zeros(1)
        self.theta = self.generate_theta(xdim)
        self.decision = decision
        self.verbose = verbose
        self.t0 = t0
        self.t1 = t1
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch = 25

    # ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # ...
        self.theta = self.generate_theta(X.shape[1], 1)
        if self.verbose:
            print('fit(...)')
            print('X.shape : ', X.shape)
            print('y.shape : ', y.shape)
            print('theta.shape : ', self.theta.shape)

        # ...
        self.theta, self.cost_history = self.stochastic_gradient_descent_momentum(self.theta, X, y, epochs=self.epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = self.hypothesis(X, self.theta)
        y[y >= self.decision] = 1
        y[y < self.decision] = 0
        return y.reshape(len(y))

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # calculate accuracy
        return (y_true == y_pred).mean()


    # ...
    def generate_theta(self, xdim=3, ydim=1):
        # if os.path.isfile(MODEL_PATH):
        #     self.load(MODEL_PATH)
        #     return self.theta
        return np.random.randn(xdim, ydim)

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        '''
        For large positive values of x, the sigmoid should be close to 1, while for large negative values, the sigmoid should be close to 0. 
        Evaluating sigmoid(0) should give you exactly 0.5.

        this method work with vectors and matrices. For a matrix, method perform the sigmoid function on every element.

        Sigmoid function formula
        Math :
            g(z) = (1) / (1 + e^(-z))

        LaTex :
            g(z) = \frac{1}{1 + e^{-z}}
        '''
        return 1.0 / ( 1.0 + np.exp(-X))

    def hypothesis(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        '''
        calculate the cost for given X and y, the following shows and example of a single dimensional X
        theta   = Vector of theta;
        X       = Row of X's np.zeros((m, j));

        where:
            m : number of samples;
            j : is the no. of features;

        Return ...........    
        

        logistic regression hypothesis formula
        Math :
            1. h(x) = g(theta^T * x); g is sigmoid function.
            2. g(z) = (1) / (1 + e^(-z))

        LaTex :
            1. h_{\theta} (x) = g( \theta^{T} x)
            2. g(z) = \frac{1}{1 + e^{-z}}
        '''
        return self.sigmoid(np.dot(X, theta))

    def cost_function(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        calculate the cost for given X and y, the following shows and example of a single dimensional X
        theta   = Vector of theta;
        X       = Row of X's np.zeros((m, j));
        y       = Actual of y's np.zeros((m, 1));

        where:
            m : number of samples;
            j : is the no. of features;
        
        Return ...........    
            
        Cost function formula
        Math :
            J = (-1. / m) * sum(y .* log(h) + (1 - y) .* log(1 - h))
        LaTex:
            J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}log(h_{\theta}(x^{(i)})) + (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))]

        Extra info.
        use np.nan_to_num() for np.log(1 - h) in case (1-h) == 0
        '''
        try:
            m = len(y)
            h = self.hypothesis(X, theta)

            cost_class_1 = np.multiply(y, np.ma.log(h).filled(1))
            cost_class_2 = np.multiply(1 - y, np.ma.log(1 - h).filled(1))  # fill with one on case we pass 0 to log(), and one will save (1-y)

            # regularization ...
            L = (self.lambda_ / (2. * m)) * (theta ** 2).sum()
            return (-1. / m) * np.sum(cost_class_1 + cost_class_2) + L
        except Exception as err:
            print('ERR > cost_function(...)  ==>  {err}.'.format(err=err))
            return 0.

    def gradient_function(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        X    = Matrix of X with added bias units;
        y    = Vector of Y;
        theta=Vector of thetas np.random.randn(j,1);

        Return ...........    

        Math:
            grad = (1 / m) * sum((h - y) .* X)

        LaTex:
            \frac{\partial J(\theta)}{\partial \theta_{j}} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}
        '''
        try:
            # ...
            m = len(y)
            h = self.hypothesis(X, theta)

            # Regularization ...
            theta_excluding_zero = np.c_[0, theta[1:].T].T
            L = (self.lambda_ / m) * theta_excluding_zero

            return (1. / m) * X.T.dot(h-y) + L
        except Exception as err:
            print('ERR > gradient_function(...)  ==>  {err}.'.format(err=err))
            return theta


    # ...
    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def learning_schedule(self, t):
        return self.t0 / (self.t1 + t)

    def stochastic_gradient_descent_momentum(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, epochs: int) -> np.ndarray:
        try:
            # ...
            print('Start SGD Momentum')
            cost_history = np.zeros(epochs)
            v = np.zeros(theta.shape)
            m = len(y)

            # ...
            for epoch in range(epochs):
                if self.verbose:
                    print('='*50)
                    print('epoch [{}]'.format(epoch + 1))
                X, y = self._shuffle(X, y)
                cost = np.zeros(m)

                # ...
                for i in range(m):
                    p = (i/m) * 100 + 1
                    if self.verbose:
                        sys.stdout.write("step progress: %d%%   \r" % p)
                        sys.stdout.flush()

                    # ...
                    random_index = np.random.randint(m)
                    xi = X[random_index: random_index+1]
                    yi = y[random_index: random_index+1]

                    # momentum ...
                    gradients = self.gradient_function(theta, xi, yi)
                    v = self.gamma * v + self.learning_schedule(epochs * m + i) * gradients
                    # v = self.gamma * v + self.alpha * gradients
                    # v = self.gamma * v + (self.alpha / (epoch + 1)) * gradients
                    theta -= v

                    # ...
                    cost[i] = self.cost_function(theta, xi, yi)
                    # break

                # print('gradient.shape : ', gradients.shape)
                # print('theta.shape : ',theta.shape)
                print('\nCost : ', cost.mean())

                # ...
                cost_history[epoch] = cost.mean()
                # break

            return theta, cost_history
        except Exception as err:
            print('ERR > stochastic_gradient_descent(..)  ==>  {err}.'.format(err=err))
            return theta, []


    # ...
    def save(self, name: str) -> None:
        np.savetxt(name, self.theta, delimiter=',')

    def load(self, name: str) -> None:
        t = np.loadtxt(name, delimiter=',')
        self.theta = t.reshape(len(t), 1)


    # ...
    def plot_error(self):
        plt.plot(range(1, len(self.cost_history)+1), self.cost_history)
        plt.title('logistic regression cost, epochs:{epochs} | t0:{t0} | t1:{t1} | gamma:{gamma}| lambda:{lambda_}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.grid()
        plt.show()


    def cost_function_loop(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            m = len(y)
            cost = 0.
            for xx, yy in zip(X, y):
                h = self.hypothesis(xx, theta)

                cost_class_1 = np.multiply(yy, np.ma.log(h).filled(1))
                cost_class_2 = np.multiply(1 - yy, np.ma.log(1 - h).filled(1))  # fill with one on case we pass 0 to log(), and one will save (1-y)
                cost += (cost_class_1 + cost_class_2)

            # TODO ...
            # self.lambda_ = 0.1
            # L = (self.lambda_ / 2. * m) * (theta ** 2).sum()
            return (-1. / m) * cost
        except Exception as err:
            print('ERR > cost_function(...)  ==>  {err}.'.format(err=err))
            return 0.

 

class MNIST:
    def __init__(self):
        ## ZIP
        self.ZIP_TRAIN_DATA = '../data/train-images-idx3-ubyte.gz'
        self.ZIP_TRAIN_LABEL = '../data/train-labels-idx1-ubyte.gz'
        self.ZIP_VALIDATION_DATA = '../data/t10k-images-idx3-ubyte.gz'
        self.ZIP_VALIDATION_LABEL = '../data/t10k-labels-idx1-ubyte.gz'

        ## File
        self.TRAIN_DATA = '../data/train-images-idx3-ubyte'
        self.TRAIN_LABEL = '../data/train-labels-idx1-ubyte'
        self.VALIDATION_DATA = '../data/t10k-images-idx3-ubyte'
        self.VALIDATION_LABEL = '../data/t10k-labels-idx1-ubyte'

    def get_data(self):
        X, y = self.get_train_data()
        Xx, yy = self.get_validation_data()
        return X, y, Xx, yy

    def get_train_data(self):
        return idx2numpy.convert_from_file(self.TRAIN_DATA), idx2numpy.convert_from_file(self.TRAIN_LABEL)

    def get_validation_data(self):
        return idx2numpy.convert_from_file(self.VALIDATION_DATA), idx2numpy.convert_from_file(self.VALIDATION_LABEL)


#################
#
#################

def plot_every_digit(X,y):
    for i in range(10):
        idx = np.where(y == i)[0][0]
        print(idx)

        digit = X[idx].reshape(X.shape[1:])
        # plt.imshow(digit)
        plt.imshow(digit, cmap = matplotlib.cm.binary, interpolation='nearest')
        plt.title(y[idx])
        plt.axis('off')
        plt.show()

def get_prime_number(y):
    prime_number = [2, 3, 5, 7]
    not_prime_number = [4, 6, 8]
    mask_prime_number = np.isin(y, prime_number)
    y[mask_prime_number] = 0
    y[y != 0] = 1
    return y

def remove_zero_and_one(X, y):
    i = np.where((y == 0) | (y == 1))
    print('To remove: ', i)
    return np.delete(X, i, axis=0), np.delete(y, i)


class LearningCurve():
    def ___ini__(self, t0=5, t1=50, gamma=0.9, lambda_=0.1, epochs=5, threshold=3000):
        self.t0 = t0
        self.t1 = t1
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epochs = epochs
        self.threshold = threshold

    def learning_curve_data_size(self, X, y, X_val, y_val):
        # ...
        m = len(X)
        axis = []
        error_train = []
        error_val = []
        itr = 0

        for i in range(self.threshold, m, self.threshold):
            # ...
            itr +=1
            print('>>>> itr: ', itr)
            axis.append(i)
            self.lambda_ = 0.1

            # ...
            logistic_regression = Model(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_)
            logistic_regression.fit(X[:i], y[:i])

            # ...
            logistic_regression.lambda_ = 0.
            theta = logistic_regression.theta
            error_train.append(logistic_regression.cost_function_loop(theta, X[:i], y[:i])[0] )
            error_val.append(logistic_regression.cost_function_loop(theta, X_val, y_val)[0] )

        return axis, error_train, error_val

    def plot_learning_curve_data_size(self, axis, error_train, error_val):
        plt.plot(axis, error_train, c='red', label='train')
        plt.plot(axis, error_val, c='blue', label='validation')
        for x, te, ve in zip(axis, error_train, error_val):
            plt.annotate(
                '{}'.format(round(te, 2)),
                (x, te),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
            plt.annotate(
                '{}'.format(round(ve, 2)),
                (x, ve),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )

        plt.title('learning curve loss, epochs {epochs}/by model| t0:{t0}/t1:{t1} | gamma:{gamma} | lambda:{lambda_}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_))
        plt.xlabel('size of data')
        plt.ylabel('error',rotation=90)
        plt.legend()
        plt.grid()
        plt.show()



    def learning_curve_acc(self, X, y, X_val, y_val):
        # ...
        m = len(X)
        axis = []
        error_train = []
        error_val = []
        itr = 0

        for i in range(self.threshold, m, self.threshold):
            # ...
            itr +=1
            print('>>>> itr: ', itr)
            axis.append(i)

            # ...
            logistic_regression = Model(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_)
            logistic_regression.fit(X[:i], y[:i])

            # ...
            y_pred = logistic_regression.predict(X[:i])
            error_train.append(Model.evaluate(y[:i], y_pred) * 100)

            y_pred = logistic_regression.predict(X_val[:i])
            error_val.append(Model.evaluate(y_val[:i], y_pred) * 100)

        return axis, error_train, error_val

    def plot_learning_curve_acc(self, axis, error_train, error_val):
        plt.plot(axis, error_train, c='red', label='train')
        plt.plot(axis, error_val, c='blue', label='validation')
        for x, te, ve in zip(axis, error_train, error_val):
            plt.annotate(
                '{} %'.format(round(te, 2)),
                (x, te),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
            plt.annotate(
                '{} %'.format(round(ve, 2)),
                (x, ve),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
        plt.title('learning curve acc, epochs {epochs}/by model | t0:{t0}/t1:{t1} | gamma:{gamma} | lambda:{lambda_}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_))
        plt.xlabel('size of data')
        plt.ylabel('accuracy (%)',rotation=90)
        plt.legend()
        plt.grid()
        plt.show()



    def learning_curve_epoch(self, X, y, X_val, y_val):
        # ...
        m = len(X)
        axis = [0]
        error_train = [0]
        error_val = [0]
        itr = 0

        logistic_regression = Model(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_)

        for epoch in range(10, 110, 10):
            # ...
            itr +=1
            print('>>>> itr: ', itr)
            print(axis)
            print(error_train)
            print(error_val)
            axis.append(axis[-1] + self.epochs)

            # ...
            logistic_regression.fit(X, y)

            # ...
            y_pred = logistic_regression.predict(X)
            error_train.append(Model.evaluate(y, y_pred) * 100)

            y_pred = logistic_regression.predict(X_val)
            error_val.append(Model.evaluate(y_val, y_pred) * 100)


        return axis, error_train, error_val

    def plot_learning_curve_epoch(self, axis, error_train, error_val):
        plt.plot(axis, error_train, c='red', label='train')
        plt.plot(axis, error_val, c='blue', label='validation')
        for x, te, ve in zip(axis, error_train, error_val):
            plt.annotate(
                '{} %'.format(round(te, 2)),
                (x, te),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
            plt.annotate(
                '{} %'.format(round(ve, 2)),
                (x, ve),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
        plt.title('learning curve acc, t0:{t0}/t1:{t1}| gamma:{gamma} | lambda:{lambda_}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_))
        plt.xlabel('epoch')
        plt.ylabel('accuracy (%)',rotation=90)
        plt.legend()
        plt.grid()
        plt.show()



    def learning_curve_gamma(self, X, y, X_val, y_val):
        # ...
        m = len(X)
        axis = []
        error_train = []
        error_val = []
        itr = 0

        for g in range(1, 11):
            # ...
            itr +=1
            print('>>>> itr: ', itr)
            print(axis)
            print(error_train)
            print(error_val)
            axis.append(self.gamma * g)

            # ...
            logistic_regression = Model(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma * g)
            logistic_regression.fit(X, y)

            # ...
            y_pred = logistic_regression.predict(X)
            error_train.append(Model.evaluate(y, y_pred) * 100)

            y_pred = logistic_regression.predict(X_val)
            error_val.append(Model.evaluate(y_val, y_pred) * 100)

        return axis, error_train, error_val

    def plot_learning_curve_gamma(self, axis, error_train, error_val):
        plt.plot(axis, error_train, c='red', label='train')
        plt.plot(axis, error_val, c='blue', label='validation')
        for x, te, ve in zip(axis, error_train, error_val):
            plt.annotate(
                '{}'.format(round(te, 1)),
                (x, te),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
            plt.annotate(
                '{}'.format(round(ve, 1)),
                (x, ve),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
        plt.title('learning curve acc, epochs:{epochs} | t0:{t0}/t1:{t1}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma))
        plt.xlabel('gamma')
        plt.ylabel('accuracy (%)',rotation=90)
        plt.legend()
        plt.grid()
        plt.show()




    def learning_curve_lambda(self, X, y, X_val, y_val):
        # ...
        m = len(X)
        axis = []
        error_train = []
        error_val = []
        itr = 0

        for g in range(0, 4):
            # ...
            itr +=1
            print('>>>> itr: ', itr)
            print(axis)
            print(error_train)
            print(error_val)
            axis.append(self.lambda_ * g)

            # ...
            logistic_regression = Model(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma, lambda_=self.lambda_ * g)
            logistic_regression.fit(X, y)

            # ...
            y_pred = logistic_regression.predict(X)
            error_train.append(Model.evaluate(y, y_pred) * 100)

            y_pred = logistic_regression.predict(X_val)
            error_val.append(Model.evaluate(y_val, y_pred) * 100)


        return axis, error_train, error_val

    def plot_learning_curve_lambda(self, axis, error_train, error_val):
        plt.plot(axis, error_train, c='red', label='train')
        plt.plot(axis, error_val, c='blue', label='validation')
        for x, te, ve in zip(axis, error_train, error_val):
            plt.annotate(
                '{}'.format(round(te, 1)),
                (x, te),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
            plt.annotate(
                '{}'.format(round(ve, 1)),
                (x, ve),
                xytext = (3, 3),
                textcoords = 'offset points',
                ha = 'left',
                va = 'top'
            )
        plt.title('learning curve acc, epochs:{epochs} | t0:{t0}/t1:{t1} | gamma:{gamma}'.format(epochs=self.epochs, t0=self.t0, t1=self.t1, gamma=self.gamma))
        plt.xlabel('lambda')
        plt.ylabel('accuracy (%)',rotation=90)
        plt.legend()
        plt.grid()
        plt.show()




def learning_curve_decision(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    # ...
    m = len(X)
    axis = []
    error_train = []
    error_val = []
    recalls = []
    precisions = []
    threshold = 3000
    itr = 0

    alpha = 0.5
    t0 = 5
    t1 = 50
    gamma = 0.9
    lambda_ = 0.1
    epochs = 5
    decision = 0.1


    for g in range(4, 9):
        # ...
        itr +=1
        print('>>>> itr: ', itr)
        axis.append(round(decision * g, 2))

        # ...
        logistic_regression = Model(epochs=epochs, t0=t0, t1=t1, gamma=gamma, lambda_=lambda_)
        logistic_regression.decision = decision * g
        logistic_regression.fit(X, y)

        # ...
        y_pred = logistic_regression.predict(X)
        error_train.append(Model.evaluate(y, y_pred) * 100)

        # ...
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recalls.append(recall)
        precisions.append(precision)

        # ...
        y_pred = logistic_regression.predict(X_val)
        error_val.append(Model.evaluate(y_val, y_pred) * 100)

        print(axis)
        print(error_train)
        print(error_val)
        print(precisions)
        print(recalls)

    return axis, error_train, error_val, recalls, precisions

def plot_learning_curve_decision(axis, error_train, error_val):
    plt.plot(axis, error_train, c='red', label='train')
    plt.plot(axis, error_val, c='blue', label='validation')
    for x, te, ve in zip(axis, error_train, error_val):
        plt.annotate(
            '{}'.format(round(te, 1)),
            (x, te),
            xytext = (3, 3),
            textcoords = 'offset points',
            ha = 'left',
            va = 'top'
        )
        plt.annotate(
            '{}'.format(round(ve, 1)),
            (x, ve),
            xytext = (3, 3),
            textcoords = 'offset points',
            ha = 'left',
            va = 'top'
        )
    plt.title('learning curve acc, epochs:{epochs} | t0:{t0}/t1:{t1} | gamma:{gamma} | lambda:{lambda_}'.format(epochs=10, t0=5, t1=50, gamma=0.9, lambda_=0.1))
    plt.xlabel('decision')
    plt.ylabel('accuracy (%)',rotation=90)
    plt.legend()
    plt.grid()
    plt.show()





def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions, "b--", label="Precision")
    plt.plot(thresholds, recalls, "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    # plt.ylim([0, 1])
    plt.grid()
    plt.show()


@timer
def main():
    # DATA ...
    minst = MNIST()
    X, y = minst.get_train_data()
    X_val, y_val = minst.get_validation_data()
    # plot_every_digit(X, y)

    # ...
    X = X.copy()
    # y_ = y.copy()
    y = y.copy()

    # ...
    X_val = X_val.copy()
    y_val = y_val.copy()
    # y_val_ = y_val.copy()



    # Reshape ...
    X_val = np.reshape(X_val, (len(X_val), X_val.shape[1] * X_val.shape[2] ))
    X_val = np.hstack([np.ones([len(X_val), 1]), X_val])

    X = np.reshape(X, (len(X), X.shape[1] * X.shape[2] ))
    X = np.hstack([np.ones([len(X), 1]), X])
    print('X: ', X.shape)
    print('y: ', y.shape)

    # Remove 0 and 1...
    X, y = remove_zero_and_one(X, y)
    X_val, y_val = remove_zero_and_one(X_val, y_val)
    print('X: ', X.shape)
    print('y: ', y.shape)

    # Prime number ...
    y = get_prime_number(y)
    y_val = get_prime_number(y_val)
    print('X: ', X.shape)
    print('y: ', y.shape)

    # scaling
    print('max ', X.max())
    print('max ', X_val.max())
    X = X / X.max()
    X_val = X_val / X_val.max()
    print('max ', X.max())
    print('max ', X_val.max())
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('X_val: ', X_val.shape)
    print('y_val: ', y_val.shape)



    # # ...
    logistic_regression = Model(epochs=10, t0=5, t1=50,  gamma=0.9, lambda_=0.1, decision=0.5)
    logistic_regression.fit(X, y)
    logistic_regression.plot_error()
    logistic_regression.save(MODEL_PATH)

    # # # ...
    # MODEL_PATH = './models/MINIST_PRIME_NUMBERS_epoch_10_gamma_09_lambda_0.out'
    # logistic_regression.load(MODEL_PATH)


    # # # # ...
    y_pred = logistic_regression.predict(X)

    a = Model.evaluate(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision_, recall_, threshold_ = precision_recall_curve(y, y_pred)


    print('train acc: {} %'.format(round(a, 2)))
    print('train recall: ', recall)
    print('train precision: ', precision)
    print('train f1: ', f1)
    print('train confusion_matrix:\n', cm)

    # ...
    y_pred = logistic_regression.predict(X_val)
    a = Model.evaluate(y_val, y_pred)
    print('test acc: {} %'.format(round(a, 2)))



    ## ...
    # axis, error_train, error_val = learning_curve_acc(X, y, X_val, y_val)
    # plot_learning_curve_acc(axis, error_train, error_val)

    # axis, error_train, error_val = learning_curve_data_size(X, y, X_val, y_val)
    # plot_learning_curve_data_size(axis, error_train, error_val)

    # axis, error_train, error_val = learning_curve_epoch(X, y, X_val, y_val)
    # # axis = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # # error_train = [0, 88.23492130558783, 88.26661033062216, 88.14830463716066, 87.99830991866484, 88.10605260378156, 88.05323756205767, 88.12717862047111, 88.19689447554664, 88.15675504383648]
    # # error_val = [0, 89.22003804692454, 89.20735573874445, 89.24540266328472, 88.91566265060241, 89.11857958148383, 89.09321496512365, 89.32149651236526, 89.24540266328472, 89.19467343056436]
    # plot_learning_curve_epoch(axis, error_train, error_val)

    # axis, error_train, error_val = learning_curve_gamma(X, y, X_val, y_val)
    # plot_learning_curve_gamma(axis, error_train, error_val)

    # axis, error_train, error_val = learning_curve_lambda(X, y, X_val, y_val)
    # plot_learning_curve_lambda(axis, error_train, error_val)

    # axis, error_train, error_val, recalls, precisions = learning_curve_decision(X, y, X_val, y_val)
    # axis = [0.4, 0.5, 0.6000000000000001, 0.7000000000000001, 0.8, 0.9]
    # error_train = [85.66599767613816, 88.21379528889828, 87.81028837012781, 82.0703496355762, 73.37488116615613, 57.908524347734236]
    # error_val = [86.83576410906785, 89.18199112238428, 89.01712111604311, 83.1705770450222, 75.20608750792644, 58.31325301204819]
    # plot_learning_curve_decision(axis, error_train, error_val)


    # axis = [0.4, 0.5, 0.6, 0.7]
    # recalls = [0.8000717102904267, 0.8601415473701142, 0.9187933923868805, 0.9538808334348726]
    # precisions = [0.9471137521222411, 0.9078947368421053, 0.8144736842105263, 0.6645585738539899]

    # plot_precision_recall_vs_threshold(precisions, recalls, axis)


MODEL_PATH = './models/MINIST_PRIME_NUMBERS.out'

if __name__ == '__main__':
    print('========================================== START ==========================================')
    #...
    main()
    print('========================================== END ============================================')