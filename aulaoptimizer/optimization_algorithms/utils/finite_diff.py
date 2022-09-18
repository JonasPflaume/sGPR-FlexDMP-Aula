import numpy as np


def finite_diff_grad(f, x, eps):
    n = x.size
    grad = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        grad[i] = (f(x + ei) - f(x - ei)) / (2 * eps)
    return grad


def finite_diff_hess(f, x, eps):
    """
    Arguments:
    ----
    f: function
    x: np.array 1-D
    eps: float

    Returns:
    ----
    hess: np.array 2-D

    """
    n = x.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            hess[i, j] = (f(x + ei + ej) + f(x - ei - ej)
                          - f(x + ei - ej) - f(x - ei + ej)) / (4 * eps * eps)
            if i != j:
                hess[j, i] = hess[i, j]
    return hess


def check_mathematical_program(fun, x, eps, verbose=False):
    """
    Input:
    fun: ( np.array 1d , np.array 2d ) = fun( np.array 1d)
    x: np.array, 1d
    eps: float

    Returns
    ----
    flags: bool ( J == J_num )
    J: np.array 2d
    J_num: np.array 2d

    """
    n = x.size
    phi, J = fun(x)
    J_num = np.zeros_like(J)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        J_num[:, i] = (fun(x + ei)[0] - fun(x - ei)[0]) / (2 * eps)
    if verbose:
        print("J\n{}".format(J))
        print("J_num\n{}".format(J_num))
        print("J - J_num \n{}".format(J - J_num))
    return np.allclose(J, J_num, atol=10 * eps), J, J_num
