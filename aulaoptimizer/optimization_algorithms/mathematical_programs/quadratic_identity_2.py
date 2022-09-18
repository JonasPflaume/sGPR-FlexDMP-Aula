import numpy as np

from ..interface.mathematical_program import MathematicalProgram


class QuadraticIdentity2 (MathematicalProgram):
    """
    x in R^n , with n=2
    f =  .5 x^T x
    sos = []
    eq = []
    ineq = []
    bounds = ( [ -inf , -inf], [ inf, inf] )
    """

    def evaluate(self, x):
        """
        """
        return np.array([.5 * np.dot(x, x)]), x.reshape(1, -1)

    def getDimension(self):
        """
        """
        return 2

    def getFHessian(self, x):
        """
        """
        return np.eye(2)

    def getInitializationSample(self):
        """
        """
        return np.ones(2)

    def report(self, verbose):
        """
        """
        strOut = "2d Quadratic function, Identity Hessian"
        return strOut
