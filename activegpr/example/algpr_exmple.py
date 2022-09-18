# The same example as in concept_illutration.py, utilizing the numerical gradient solver aula
import numpy as np
import matplotlib.pyplot as plt
from algpr.gpr import GaussianProcessRegressor
from algpr.kernels import RBF

# generate the data
X_full = np.linspace(-3,3,100)
Y_full = np.sin(X_full) + np.random.randn(len(X_full)) * 0.01

index = np.arange(len(X_full))
index = np.random.choice(index, 2)

X = X_full[index]
Y = Y_full[index]
old_length_scale = [10.]
for _ in range(50):
    # ALGPR
    ##############################################
    gpr = GaussianProcessRegressor(kernel=RBF(l=old_length_scale, anisotropic=False), noise_level=0.01)
    gpr.fit(X, Y, call_hyper_opt=True)
    old_length_scale = gpr.kernel.length_scale
    next_point = gpr.max_entropy_x([-3.], [3.])
    ###############################################
    plt.figure(figsize=[15,10])
    t = np.linspace(-4,4,100)
    mu, variance = gpr.predict(t, return_var=True)
    plt.plot(t, mu, label="mean prediction")
    plt.plot(t, np.sin(t), '-c', linewidth=2.5, label="Ground trueth")
    plt.plot(t, mu + variance, '-.r')
    plt.plot(t, mu - variance, '-.r', label="1 standard variance")
    plt.xlabel("input value")
    plt.ylabel("prediction")
    print(next_point)
    for i, x in enumerate(X):
        if i == 0:
            y = Y[i]
            plt.scatter(x, y, c='b',label="data points")
            plt.scatter(next_point, 0, c='r', marker='X', s=200, label="next sampling spot")
        else:
            y = Y[i]
            plt.scatter(x, y, c='b')
            plt.scatter(next_point, 0, c='r', marker='X', s=200)

    # update dataset
    ##############################################
    X = np.append(X, next_point)
    Y = np.append(Y, np.sin(next_point))
    ##############################################
    plt.legend()
    plt.show()