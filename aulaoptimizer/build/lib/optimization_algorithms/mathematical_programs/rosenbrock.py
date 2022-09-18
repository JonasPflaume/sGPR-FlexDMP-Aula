import sys
import numpy as np

from ..interface.mathematical_program import MathematicalProgram
from ..interface.objective_type import OT


class Rosenbrock(MathematicalProgram):

    """
    x = [ x_1, x_2  ] in R^2
    f =  ( a - x ) ** 2 + b * ( y - x^2 ) ^ 2
    sos = []
    eq = []
    ineq = []
    bounds = ( [ -inf , -inf], [ inf, inf] )
    """

    def __init__(self, a, b):
        """
        a: float
        b: float
        """
        self.a = a
        self.b = b
        super().__init__()

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """
        f = (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2
        phi = np.array([-2 * (self.a - x[0]) + 2 * self.b * (x[1] -
                                                             x[0]**2) * (-2 * x[0]), self.b * 2 * (x[1] - x[0]**2)])
        return np.array([f]), phi.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 2

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.array([1, 1])

    def getFHessian(self, x):
        """
        Ref: https://www.wolframalpha.com/input/?i=hessian+of+++%28+a+-+x+%29+%5E+2+%2B+b+%28+y+-+x%5E2+%29+%5E+2

        See Also
        ------
        MathematicalProgram.getFHessian
        """
        n = self.getDimension()
        a = self.a
        b = self.b
        y = x[1]
        x = x[0]
        return np.array([[- 4 * b * (y - x**2) + 8 * b * x ** 2 + 2, -4 * b * x],
                         [-4 * b * x, 2 * b]])

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Rosenbrock Function"
        return strOut
