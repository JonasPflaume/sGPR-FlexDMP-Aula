import numpy as np
import unittest
import sys
import math
sys.path.append("..")

import sys
import unittest
import numpy as np
import sys

sys.path.append("..")
from utils.finite_diff import *
from interface.mathematical_program import MathematicalProgram
from quadratic_program import QuadraticProgram as Problem


class testHs071(unittest.TestCase):
    """
    test class Constrained0
    """
    problem = Problem

    # def testOptimum(self):
    #     # n = 4
    #     x = np.array([1.00000000,4.74299963,3.82114998,1.37940829])
    #     problem = self.problem()
    #     phi, J = problem.evaluate(x)
    #     print("optimum")
    #     print("phi ", phi)
    #     print("types ", problem.getFeatureTypes())

    #     x = np.array([1.44946879,4.99999999,3.44961177,0.99997827])
    #     phi, J = problem.evaluate(x)
    #     print("non optimum")
    #     print("phi ", phi)
    #     print("types ", problem.getFeatureTypes())

    # self.assertTrue(np.allclose(phi, np.zeros(n + 1)))

    def with_all(self):
        """
        """
        n = 4
        meq = 2
        mineq = 1
        ub = []
        lb = []
        _H = np.random.rand(n, n)
        H = np.transpose(_H) @ _H
        g = np.random.rand(n)
        Aeq = np.random.rand(meq, n)
        beq = np.random.rand(meq)
        Aineq = np.random.rand(mineq, n)
        bineq = np.random.rand(mineq)
        return self.problem(H, g, Aineq, bineq, Aeq, beq, lb, ub)

    def testValue1(self):
        # n = 4
        problem = self.with_all()
        x = np.random.rand(problem.getDimension())
        phi, J = problem.evaluate(x)
        # self.assertTrue(np.allclose(phi, np.zeros(n + 1)))

    def testJacobian(self):

        problem = self.with_all()
        x = np.random.rand(problem.getDimension())
        flag, _, _ = check_mathematical_program(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testHessain(self):

        problem = self.with_all()
        x = np.random.rand(problem.getDimension())

        H = problem.getFHessian(x)

        # we know that in this implementation, the first term is the
        # OT.f cost
        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        print("H {}".format(H))
        print("Hdiff {}".format(H))
        self.assertTrue(flag)

# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log


if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()
