import sys
import numpy as np
import math

try:
    from ..interface.mathematical_program import MathematicalProgram
    from ..interface.objective_type import OT
except:
    from interface.mathematical_program import MathematicalProgram
    from interface.objective_type import OT


try:
    from .logistic import Logistic
except:
    from logistic import Logistic


class LogisticWithBounds(MathematicalProgram):

    """
    """

    def __init__(self):
        """
        """

        # self.K = 1.0
        # self.r = 10.0
        # self.t0 = .5

        # INIT SAMPLE = np.array([1, 1, 1])

        self.unconstrained = Logistic()
        self.UB = 2 * np.ones(3)
        self.LB = 0 * np.ones(3)

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """
        phi, j = self.unconstrained.evaluate(x)
        # x <= UB
        ub = x - self.UB
        # -x <= -LB
        lb = -x + self.LB
        return np.concatenate((phi, ub, lb)), np.vstack((j, np.identity(3), -1 * np.identity(3)))

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.unconstrained.getDimension()

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return self.unconstrained.getFeatureTypes() + 6 * [OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.unconstrained.getInitializationSample()

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Logistic Regression"
        return strOut
