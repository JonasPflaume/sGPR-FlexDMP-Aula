import numpy as np

class RBF:
    ''' RBF activation function
    '''
    def __init__(self, center, width):
        self.c = center
        self.h = width

    def __call__(self, x):
        return np.exp(-self.h * (x - self.c) ** 2 )


def integrate_RK4(start, ODEFunc, dt):
    ''' RungKutta 4th order method
    '''

    h = dt

    end = start

    k_1 = ODEFunc(end)
    k_2 = ODEFunc(end + 0.5 * h * k_1)
    k_3 = ODEFunc(end + 0.5 * h * k_2)
    k_4 = ODEFunc(end + k_3 * h)

    end = end + (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h

    return end