from random import gauss
import numpy as np
from itertools import product

from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pylab as plt
from torch import index_add

def g_pdf(xi, m, P):
    pdf = np.linalg.det(2 * np.pi * P) ** (-0.5) * np.exp(-0.5 * (xi-m).T @ np.linalg.inv(P) @ (xi-m) )
    return pdf

def g_mix(xi, m_l, P_l, alpha_l):
    pdf = 0.
    for i in range(len(m_l)):
        mi = m_l[i]
        Pi = P_l[i]
        pdfi = g_pdf(xi, mi, Pi)
        pdf += alpha_l[i] * pdfi
    return pdf

m1 = np.array([3.,1.])
P1 = np.array([[1.,0.],[0.,1.]])

m2 = np.array([2.,2.])
P2 = np.array([[2.,0.],[0.,1.]])

m3 = np.array([-3.,2.])
P3 = np.array([[1.,0.],[0.,2.]])

m4 = np.array([-1.,1.])
P4 = np.array([[1.,0.],[0.,3.]])

m5 = np.array([1.,3.])
P5 = np.array([[1.,0.],[1.,1.]])

m_l = [m1, m2, m3, m4, m5]
P_l = [P1, P2, P3, P4, P5]
alpha_l = [.2, .2, .4, .4, .2]

intens = 40
X1 = np.linspace(-5,5,intens)
Y1 = np.linspace(-5,5,intens)
X1, Y1 = np.meshgrid(X1, Y1)
Z1 = np.zeros([intens,intens])
for i in range(intens**2):
    row = i // intens
    col = i % intens
    x = np.array([X1[row, col], Y1[row, col]])
    Z1[row, col] = g_mix(x, m_l, P_l, alpha_l)
    
############# learning ###########
data_len = 200
X = [np.array( [np.random.uniform(-5,5), np.random.uniform(-5,5)] ) for i in range(data_len)]
X = np.array(X)
Y = np.zeros(data_len)
for i in range(data_len):
    Y[i] = g_mix(X[i], m_l, P_l, alpha_l)
    
from algpr.gpr import GaussianProcessRegressor
from algpr.kernels import RBF
gpr = GaussianProcessRegressor(kernel=RBF(l=[1.,1.], anisotropic=True))
gpr.fit(X, Y, call_hyper_opt=True)

Z2 = np.zeros([intens,intens])
for i in range(intens**2):
    row = i // intens
    col = i % intens
    x = np.array([X1[row, col], Y1[row, col]]).reshape(1,2)
    Z2[row, col] = gpr.predict(x)
    
x0 = np.array([0.,0]).reshape(-1,2)
dist = gpr.kernel(x0, X).squeeze()
choosen_num = 15
index_ball = np.argsort(dist)[-choosen_num:]
choosenX = X[index_ball]
index_out_ball = np.argsort(dist)[:-choosen_num]
notX = X[index_out_ball]
plt.plot(x0[0, 0],x0[0, 1], 'ro')
plt.plot(choosenX[:,0], choosenX[:,1], 'g.')
plt.plot(notX[:,0], notX[:,1], 'y.')
plt.show()
##############

# Plot the surface.
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

surf = ax1.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

surf2 = ax2.plot_surface(X1, Y1, Z2, cmap=cm.Greys,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
