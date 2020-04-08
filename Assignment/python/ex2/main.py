import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sal_timer import timer


def get_data():
    df = pd.read_csv(DATA_PATH)
    print(df.head(5))

    return df

def scatter_training_data(df):
    plt.plot('exam1', 'exam2', '*', c='r', data=df[df.y == 0])
    plt.plot('exam1', 'exam2', '+', c='b', data=df[df.y == 1])
    plt.legend(['Not Admitted', 'Admitted'])
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Scatter plot of training data')
    plt.show()


def sigmoid(X: np.ndarray) -> np.ndarray:
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

def hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    '''
    calculate the cost for given X and y, the following shows and example of a single dimensional X
    theta   = Vector of theta;
    X       = Row of X's np.zeros((m, j));

    where:
        m number of samples;
        j is the no. of features;

    Return ...........    
    

    logistic regression hypothesis formula
    Math :
        1. h(x) = g(theta^T * x); g is sigmoid function.
        2. g(z) = (1) / (1 + e^(-z))

    LaTex :
        1. h_{\theta} (x) = g( \theta^{T} x)
        2. g(z) = \frac{1}{1 + e^{-z}}
    '''

    # https://www.youtube.com/watch?v=okpqeEUdEkY
    # np.dot(theta.T, X)

    return sigmoid(np.dot(X, theta))

def cost_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    '''
    calculate the cost for given X and y, the following shows and example of a single dimensional X
    theta   = Vector of theta;
    X       = Row of X's np.zeros((m, j));
    y       = Actual of y's np.zeros((m, 1));

    where:
        m number of samples;
        j is the no. of features;
    
    Return ...........    
        
    Cost function formula
    Math :
        J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h))
    LaTex:
        J(\theta) = \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)}log(h_{\theta}(x^{(i)})) - (1 - y^{(i)})log(1 - h_{\theta}(x^{(i)}))]
    '''

    try:
        m = len(y)
        h = hypothesis(X, theta)
        
        # print('y : ', y.shape)
        # print('h : ', h.shape)
        # print('log(h) : ', np.log(h))
        # print(
        #     np.multiply(
        #         -y,
        #         np.log(h)
        #         ).shape
        #     )

        return (1. / m) * np.sum( np.multiply(-y, np.log(h ) - np.multiply(1 - y, np.log(1 - h)) ))
    except Exception as err:
        print('cost_function(...)  ==>  {err}.'.format(err=err))
        return 0.

def gradient_function(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        m = len(y)
        h = hypothesis(X, theta)
        return (1. / m) * np.sum(np.multiply(h - y, X), axis=0)
    except Exception as err:
        print('gradient_function(...)  ==>  {err}.'.format(err=err))
        return np.zeros(theta.shape)


# TODO ... https://www.youtube.com/watch?v=QOne3o-7_DQ
def gradient_descent(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: int, num_iterations: int) -> np.ndarray:
    '''
    X    = Matrix of X with added bias units;
    y    = Vector of Y;
    theta=Vector of thetas np.random.randn(j,1);
    learning_rate;
    iterations = no of iterations;
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    h = hypothesis(X, theta)
    theta = theta - alpha / m * np.dot((h - y), X)

    ####
    # cost_history = [] # plot
    # theta = theta + alpha * gradient


    ####
    # https://gist.github.com/sagarmainkar/41d135a04d7d3bc4098f0664fe20cf3c

    # m = len(y)
    # cost_history = np.zeros(iterations)
    # theta_history = np.zeros((iterations,2))
    # for it in range(iterations):
        
    #     prediction = np.dot(X,theta)
        
    #     theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
    #     theta_history[it,:] =theta.T
    #     cost_history[it]  = cal_cost(theta,X,y)
        
    # return theta, cost_history, theta_history



def sigmoid_test():
    x = np.array([
        99999,
        100,
        10,
        4,
        2,

        0.01,
        0.1,
        0,
        -0.1,
        -0.01,

        
        -2,
        -4,
        -10,
        -100,
        -99999,
    ])

    print(pd.DataFrame({
        'x': x,
        'sig': sigmoid(x),
        'real': np.array([1.00,1.00,0.99,0.98,0.88,0.50,0.52,0.50,0.47,0.49,0.11,0.17,0.45,0.37,0.00])
    }))

    x = 4
    z = sigmoid(x)
    print(z)

def hypothesis_test():
    X = np.array([34.62365962451697, 78.0246928153624])
    theta = np.array([1, 1])
    y = np.array([0])
    # yy = hypothesis(X, theta)
    # print(yy)

    # ...
    X = np.array([7, 2])
    theta = np.array([3, 4])
    y = hypothesis(X, theta)
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

    # ...
    X = np.array([1, 2])
    theta = np.array([3, 4]).T
    y = hypothesis(X, theta)
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

    # ...
    X = np.array([1, 2]).T
    theta = np.array([3, 4])
    y = hypothesis(X, theta)
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

    # ...
    X = np.array([[1, 2], [1, 2], [1, 2]])
    theta = np.array([3, 4])
    y = hypothesis(X, theta)
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

    # ... ERROR
    # X = np.array([[1, 2], [1, 2], [1, 2]]).T
    # theta = np.array([3, 4])
    # y = hypothesis(X, theta)
    # print('X : ', X.shape)
    # print('theta : ', theta.shape)
    print('y : ', y.shape)
    # print(y)

    # ...
    X = np.array([[1, 2], [1, 2], [1, 2]])
    theta = np.array([3, 4]).T
    y = hypothesis(X, theta)
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

    # ...
    X = np.array([[1, 2], [1, 2], [1, 2]]).T
    theta = np.array([3, 4]).T
    y = hypothesis(theta, X)  # <<===== replace in space !!!
    print('X : ', X.shape)
    print('theta : ', theta.shape)
    print('y : ', y.shape)
    print(y)

def cost_function_test():
    # ...
    X = np.array([[1, 2], [1, 2], [1, 2]])
    m, n = X.shape
    theta = np.array([[3], [4]])
    # y = np.array([1., 1., 0.])
    y = np.array([[1.], [1.], [0.]])

    # ...
    J = cost_function(theta, X, y)
    print(J)

def gradient_function_test():
    # ...
    X = np.array([[1, 2], [1, 2], [1, 2]])
    theta = np.array([[3], [4]])
    # y = np.array([1., 1., 0.])
    y = np.array([[1.], [1.], [0.]])

    # ...
    grad = gradient_function(theta, X, y)
    print(grad)


def test_1():
    # ...
    df = get_data()
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    m, n = X.shape
    X = np.hstack([np.ones([m, 1]), X])
    initial_theta = np.zeros([n + 1, 1])
    y = np.resize(y, (len(y), 1))
    # y = y.reshape((len(y), 1))

    # ...
    print('X : ', X.shape)
    print('Theta : ', initial_theta.shape)
    print('y : ', y.shape)

    # ...
    cost = cost_function(initial_theta, X, y)
    print('\n'*2)
    print('Cost at initial theta (zeros): ', cost)
    print('Expected cost (approx): 0.693')

    grad = gradient_function(initial_theta, X, y)
    print('Gradient at initial theta:\t\t', grad)
    print('Expected gradients (approx):\t\t[-0.1000 -12.0092 -11.2628]')


    test_theta = np.array([[-24], [0.2], [0.2]])

    cost = cost_function(test_theta, X, y)
    print('\n'*2)
    print('Cost at test theta: ', cost)
    print('Expected cost (approx): 0.218')

    grad = gradient_function(test_theta, X, y)
    print('Gradient at test theta:\t\t', grad)
    print('Expected gradients (approx): \t\t[0.043 2.566 2.647]')





@timer
def main():
    df = get_data()
    # scatter_training_data(df)



    
    # X, y = df.iloc[:, :-1].values, df.iloc[-1].values
    # print(X[:3])
    # print(y[:3])


@timer
def test():
    # sigmoid_test()
    # hypothesis_test()
    # cost_function_test()
    # gradient_function_test()
    test_1()



DATA_PATH = './data/ex2data1.csv'

if __name__ == '__main__':
    print('========================================== START ==========================================')
    #...
    test()
    # main()
    print('========================================== END ============================================')