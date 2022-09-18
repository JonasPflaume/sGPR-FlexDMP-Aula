import sys
import numpy as np
import math

from ..interface.mathematical_program import MathematicalProgram
from ..interface.objective_type import OT


class Constrained0(MathematicalProgram):
    """
    x = [ x_1, x_2, ... , x_n ] in R^n
    f =  SUM( x_i )
    sos = []
    eq = []
    ineq = [ x_i >= 0 ] ( implemented as  -x_i <= 0 )
    """

    def __init__(self, n):
        """
        Arguments:
        ----
        n: interger
        """
        self.n = n
        super().__init__()

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """
        n = self.getDimension()  # we can also access self.n directly
        phi = np.zeros(n + 1)
        phi[0] = np.sum(x)
        phi[1:] = -x
        J = np.zeros((n + 1, n))
        J[0] = np.ones(n)
        J[1:, :] = -np.identity(n)
        return phi, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.n

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f] + [OT.ineq] * self.getDimension()

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        n = self.getDimension()
        return np.zeros((n, n))

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Example of Constrained Problem"
        return strOut
