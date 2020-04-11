import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sal_timer import timer

def cost_fun_1(x):
    return 0.5 * x**2


def cost_fun_rmse(x, y):
    return 0.5 * (x - y)**2




@timer
def main():
    x = np.arange(-5., 5., 1.)
    y_ = np.array([12., 8.2, 4.15, 2.3, 1.5, 0., 1.5, 2.7, 4.5, 8.4])
    y = cost_fun_1(x)

    # ...
    # print(x)
    # print(y)
    # plt.plot(x,y)
    # plt.show()


    # ...
    # y = cost_fun_rmse(x, y_)
    # print(x)
    # print(y_)
    # print(y)
    # plt.plot(x,y)
    # plt.show()



if __name__ == '__main__':
    print('========================================== START ==========================================')
    #...
    main()
    print('========================================== END ============================================')