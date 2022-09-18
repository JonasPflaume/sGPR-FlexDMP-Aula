import sys
import numpy as np

try:
    from ..interface.mathematical_program import MathematicalProgram
    from ..interface.objective_type import OT
except:
    from interface.mathematical_program import MathematicalProgram
    from interface.objective_type import OT


class QuadraticProgram(MathematicalProgram):

    """
    REF https://coin-or.github.io/Ipopt/INTERFACES.html
    solution is:
    x∗=(1.00000000,4.74299963,3.82114998,1.37940829).
    """

    def __init__(self, H, g, Aineq=[], bineq=[], Aeq=[], beq=[], lb=[], ub=[]):
        """
        NOTE: H must be symmetric!!

        min .5 * x.T * H * x + g.T * x
        s.t.
        Aineq * x <= b
        Aeq * x = b
        lb <= x <= ub
        """
        self.H = H
        self.g = g
        self.Aineq = Aineq
        self.bineq = bineq
        self.Aeq = Aeq
        self.beq = beq
        self.lb = lb
        self.ub = ub

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """

        # cost
        f = 0.5 * x @ (self.H @ x) + self.g @ x
        df = self.H @ x + self.g

        phi = np.array([f])
        J = df.reshape(1, -1)

        if len(self.Aineq) > 0:
            phi = np.concatenate((phi, self.Aineq @ x - self.bineq))
            J = np.vstack((J, self.Aineq))

        if len(self.Aeq) > 0:
            phi = np.concatenate((phi, self.Aeq @ x - self.beq))
            J = np.vstack((J, self.Aeq))

        return phi, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.H.shape[0]

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f] + len(self.Aineq) * [OT.ineq] + len(self.Aeq) * [OT.eq]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFHessian(self, x):
        """
        Ref: https://www.wolframalpha.com/input/?i=hessian+of+++%28+a+-+x+%29+%5E+2+%2B+b+%28+y+-+x%5E2+%29+%5E+2

        See Also
        ------
        MathematicalProgram.getFHessian
        """
        return self.H

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Quadratic Program"
        return strOut
