import sys
import numpy as np
import math

try:
    from ..interface.mathematical_program import MathematicalProgram
    from ..interface.objective_type import OT
except:
    from interface.mathematical_program import MathematicalProgram
    from interface.objective_type import OT


class Logistic(MathematicalProgram):

    """
    """

    def __init__(self):
        """
        """
        self.K = 1.0
        self.r = 10.0
        self.t0 = .5
        self.xopt = np.array([self.K, self.r, self.t0])
        self.num_points = 10
        self.t = np.linspace(0, 1, self.num_points)
        self.data = self.y(self.t, self.xopt)

    def y(self, t, x):
        K = x[0]
        r = x[1]
        t0 = x[2]
        return K / (1.0 + np.exp(-r * (t - t0)))

    def y_d1(self, t, x):
        """
        w.r.t K
        """
        K = x[0]
        r = x[1]
        t0 = x[2]
        return 1 / (1.0 + np.exp(-r * (t - t0)))

    def y_d2(self, t, x):
        """
        w.r.t r
        """
        K = x[0]
        r = x[1]
        t0 = x[2]
        return -1 * K / (1.0 + np.exp(-r * (t - t0)))**2 * -1 * (t - t0) * np.exp(-r * (t - t0))

    def y_d3(self, t, x):
        """
        w.r.t t0
        """
        K = x[0]
        r = x[1]
        t0 = x[2]
        return -1 * K / (1.0 + np.exp(-r * (t - t0)))**2 * r * np.exp(-r * (t - t0))

    def generate_data(self, t):
        self.data = self.y(t, self.xopt)

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """
        t = self.t
        phi = self.y(t, x) - self.data
        J = np.zeros((self.num_points, 3))
        J[:, 0] = self.y_d1(t, x)
        J[:, 1] = self.y_d2(t, x)
        J[:, 2] = self.y_d3(t, x)
        return phi, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 3

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * self.num_points

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.array([1., 1., 1.])

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Logistic Regression"
        return strOut
