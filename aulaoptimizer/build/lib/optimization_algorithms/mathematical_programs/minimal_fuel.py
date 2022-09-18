import sys
import numpy as np
import math

try:
    from ..interface.mathematical_program import MathematicalProgram
    from ..interface.objective_type import OT
except:
    from interface.mathematical_program import MathematicalProgram
    from interface.objective_type import OT


class MinimumFuel(MathematicalProgram):
    def __init__(self, A, b, x_des, N):
        self.A = A
        self.b = b
        self.N = N
        n = A.shape[0]
        self.n = n
        self.x_des = x_des

        self.dim = (self.n + 1) * self.N
        self.u_inds = np.arange(self.dim)[::(self.n + 1)]

        self.H_dyn = np.zeros((self.n * self.N, self.dim))
        for t in range(N):
            cur = (n + 1) * t
            if t > 0:
                self.H_dyn[n * t:n * (t + 1), cur - n:cur] = A
            self.H_dyn[n * t:n * (t + 1), cur:cur + 1] = b
            self.H_dyn[n * t:n * (t + 1), cur + 1:cur +
                       (n + 1)] = -np.eye(self.n)

    def evaluate(self, x):

        sos = np.zeros(len(self.u_inds))
        Jsos = np.zeros((len(self.u_inds), self.dim))
        for i, ind in enumerate(self.u_inds):
            sos[i] = x[ind]
            Jsos[i, ind] = 1

        h_dyn = self.H_dyn@x
        Jh_dyn = self.H_dyn

        h_des = x[-self.n:] - self.x_des
        Jh_des = np.zeros((self.n, self.dim))
        Jh_des[:, -self.n:] = np.eye(self.n)

        phi = np.hstack([sos, h_dyn, h_des])
        J = np.concatenate([Jsos, Jh_dyn, Jh_des], axis=0)

        return phi, J

    def getFeatureTypes(self):
        return [OT.sos] * self.N + [OT.eq] * (self.n * self.N) + [OT.eq] * self.n

    def getDimension(self):
        return self.dim

    def getFHessian(self, x):
        dim = self.getDimension()
        return np.zeros((dim, dim))

    def getInitializationSample(self):
        return np.zeros(self.getDimension())
