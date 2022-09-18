import sys
import unittest
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2

from optimization_algorithms.mathematical_programs.quadratic_identity_2 import QuadraticIdentity2
from optimization_algorithms.mathematical_programs.rosenbrock import Rosenbrock
from optimization_algorithms.mathematical_programs.logistic import Logistic
from optimization_algorithms.mathematical_programs.hole import Hole


from solver import SolverUnconstrained


class testSolver(unittest.TestCase):
    """
    test SolverUnconstrained
    """
    Solver = SolverUnconstrained

    def testConstructor(self):
        """
        check the constructor
        """
        solver = self.Solver()

    def testQuadraticIdentity(self):
        problem = MathematicalProgramTraced(QuadraticIdentity2())
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.linalg.norm(np.zeros(2) - last_trace) < .01)

    def testRosenbrock(self):
        problem = MathematicalProgramTraced(Rosenbrock(2, 100))
        solver = self.Solver()
        solution = np.array([2, 4])
        solver.setProblem(problem)
        solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.allclose(last_trace, solution, 1e-2, 1e-2))

    def testHole(self):

        def make_C_exercise1(n, c):
            """
            n: integer
            c: float
            """
            C = np.zeros((n, n))
            for i in range(n):
                C[i, i] = c ** (float(i - 1) / (n - 1))
            return C

        C = make_C_exercise1(3, .1)
        problem = MathematicalProgramTraced(Hole(C, 1.5))
        solution = np.zeros(3)
        solver = self.Solver()
        solver.setProblem(problem)
        solver.solve()
        last_trace = problem.trace_x[-1]
        success = np.allclose(last_trace, solution, 1e-3, 1e-3)
        self.assertTrue(success)

    def testNonLinearSOS(self):
        problem = MathematicalProgramTraced(Logistic())
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        eps = .01
        self.assertTrue(np.linalg.norm(
            problem.mathematical_program.xopt - last_trace) < eps)


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
