import sys
import numpy as np

try:
    from ..interface.mathematical_program import MathematicalProgram
    from ..interface.objective_type import OT
except:
    from interface.mathematical_program import MathematicalProgram
    from interface.objective_type import OT


class Hs071(MathematicalProgram):

    """
    REF https://coin-or.github.io/Ipopt/INTERFACES.html
    solution is:
    x∗=(1.00000000,4.74299963,3.82114998,1.37940829).
    """

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """

        # cost
        f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        grad_f = np.zeros(4)
        grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2])
        grad_f[1] = x[0] * x[3]
        grad_f[2] = x[0] * x[3] + 1
        grad_f[3] = x[0] * (x[0] + x[1] + x[2])

        # constraint
        g = np.zeros(2)
        g[0] = 25 - x[0] * x[1] * x[2] * x[3]
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40

        Jg = np.zeros((2, 4))

        Jg[0, 0] = -x[1] * x[2] * x[3]
        Jg[0, 1] = -x[0] * x[2] * x[3]
        Jg[0, 2] = -x[0] * x[1] * x[3]
        Jg[0, 3] = -x[0] * x[1] * x[2]

        Jg[1, 0] = 2 * x[0]
        Jg[1, 1] = 2 * x[1]
        Jg[1, 2] = 2 * x[2]
        Jg[1, 3] = 2 * x[3]

        # bounds (as normal constraint)
        bU = x - 5
        bL = 1 - x

        Jb = np.zeros((8, 4))
        Jb[:4, :] = np.identity(4)
        Jb[4:, :] = -1 * np.identity(4)

        return np.concatenate((np.array([f]), g, bU, bL)), np.vstack((grad_f, Jg, Jb))

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return 4

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f, OT.ineq, OT.eq] + 8 * [OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.array([1., 5., 5., 1.])

    def getFHessian(self, x):
        """
        Ref: https://www.wolframalpha.com/input/?i=hessian+of+++%28+a+-+x+%29+%5E+2+%2B+b+%28+y+-+x%5E2+%29+%5E+2

        See Also
        ------
        MathematicalProgram.getFHessian
        """
        hess = np.zeros((4, 4))

        # we fill first lower triangular
        hess[0, 0] = 2 * x[3]  # // 0,0
        hess[1, 0] = x[3]  # // 1,0
        hess[1, 1] = 0.  # // 1,1
        hess[2, 0] = x[3]  # // 2,0
        hess[2, 1] = 0.  # // 2,1
        hess[2, 2] = 0.  # // 2,2
        hess[3, 0] = 2 * x[0] + x[1] + x[2]  # // 3,0
        hess[3, 1] = x[0]  # // 3,1
        hess[3, 2] = x[0]  # // 3,2
        hess[3, 3] = 0.  # // 3,3

        out = hess + np.transpose(hess) - np.diag(np.diag(hess))
        return out

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "hs071"
        return strOut
