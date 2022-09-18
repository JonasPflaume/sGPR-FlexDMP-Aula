import sys
import unittest
import numpy as np
import sys

sys.path.append("..")
from utils.finite_diff import *
from interface.mathematical_program import MathematicalProgram


from rosenbrock import Rosenbrock


class testProblem(unittest.TestCase):
    """
    test mathematical program Rosenbrock
    """

    problem = Rosenbrock

    def testjacobian(self):
        """
        """

        problem5 = Rosenbrock(2, 100)
        flag, _, _ = check_mathematical_program(
            problem5.evaluate, np.array([.8, -.5]), 1e-5)
        self.assertTrue(flag)

    def testoptimal(self):
        """
        """

        problem5 = Rosenbrock(2, 100)
        self.assertTrue(np.abs(problem5.evaluate(
            np.array([2, 4]))[0][0]) <= 1e-5)
        self.assertTrue(problem5.evaluate(np.array([2, 4]))[
                        0][0] <= problem5.evaluate(np.array([2, 4 + .001]))[0][0])
        self.assertTrue(problem5.evaluate(np.array([2, 4]))[
                        0][0] <= problem5.evaluate(np.array([2 + 0.001, 4]))[0][0])

    def testhessian(self):
        """
        """

        problem = Rosenbrock(2, 100)
        x = np.array([.8, -.5])

        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
