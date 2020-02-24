import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


from sal_timer import timer



def sigmoid(x):
    return 1.0 / ( 1.0 + np.exp(-x))

@timer
def main():
    
    x = np.array([
        2,
        4,
        10,
        100,

        0,

        -0.1,
        -0.44,
        -10,
        -100
    ])

    z = sigmoid(x)
    print(z)

    x = 4
    z = sigmoid(x)
    print(z)



if __name__ == '__main__':
    print('========================================== START ==========================================')
    #...
    main()
    print('========================================== END ============================================')