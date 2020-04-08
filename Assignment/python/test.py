import numpy as np
import matplotlib.pyplot as plt



def cost1(x):
    return -np.log(1. / (1. + np.exp(-x)))


def cost0(x):
    return -np.log(1. - 1. / (1. + np.exp(-x)))


# def svm1(x):
#     if x >= 1:
#         return 0
#     return np.abs(x)

def main():
    X = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    print(cost1(X))

    plt.plot(X, cost1(X))
    plt.plot(X, cost0(X))
    # plt.plot(X, [svm1(_) for _ in X])

    plt.show()



if __name__ == '__main__':
    print('========================================== START ==========================================')
    #...
    main()
    print('========================================== END ============================================')