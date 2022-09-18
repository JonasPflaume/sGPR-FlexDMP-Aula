import sys
import unittest
import numpy as np

sys.path.append("..")
from optimization_algorithms.utils.finite_diff import *
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.mathematical_programs.hs071 import Hs071
from optimization_algorithms.mathematical_programs.nonlinearA import NonlinearA
from optimization_algorithms.mathematical_programs.quadratic_program import QuadraticProgram
from optimization_algorithms.mathematical_programs.linear_program_ineq import LinearProgramIneq
from optimization_algorithms.mathematical_programs.halfcircle import HalfCircle
from optimization_algorithms.mathematical_programs.logistic_bounds import LogisticWithBounds

# from solution import SolverAugmentedLagrangian as Solver
from solver import SolverAugmentedLagrangian as Solver


class testSolver(unittest.TestCase):
    """
    test SolverUnconstrained
    """
    Solver = Solver

    def set_(self, solver):
        self.Solver = solver

    def testConstructor(self):
        """
        check the constructor
        """
        solver = self.Solver()

    def testLogisticBounds(self):
        """
        """
        problem = MathematicalProgramTraced(
            LogisticWithBounds(), max_evaluate=100000000)
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        buu = np.array([1, 2, .5])
        solution = np.array([2, 2, 1.0369])
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)
        print("solution")
        print(problem.evaluate(solution))
        print("buu")
        print(problem.evaluate(buu))

    def testQuadraticB(self):
        """
        """
        n = 3
        H = np.array([[1., -1., 1], [-1, 2, -2], [1, -2, 4]])
        g = np.array([2, -3, 1])
        Aineq = np.vstack((np.identity(n), -np.identity(n)))
        bineq = np.concatenate((np.ones(n), np.zeros(n)))
        Aeq = np.ones(3).reshape(1, -1)
        beq = np.array([.5])
        # 0 <= x <= 1
        # x <= 1
        # -x <= 0
        problem = MathematicalProgramTraced(QuadraticProgram(
            H=H, g=g, Aeq=Aeq, beq=beq, Aineq=Aineq, bineq=bineq))
        solution = np.array([0, 0.5, 0])
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)

    def testQuadraticA(self):
        """
        """
        H = np.array([[1., -1.], [-1., 2.]])
        g = np.array([-2., -6.])
        Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
        bineq = np.array([2., 2., 3.])
        problem = MathematicalProgramTraced(
            QuadraticProgram(H=H, g=g, Aineq=Aineq, bineq=bineq))
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        solution = np.array([0.6667, 1.3333])
        phi, J = problem.evaluate(solution)
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)

    def testLinearProgramIneq(self):
        """
        """
        n = 4
        problem = MathematicalProgramTraced(LinearProgramIneq(n))
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.linalg.norm(np.zeros(n) - last_trace) < .01)

    def testHalfCircle(self):
        """
        """
        problem = MathematicalProgramTraced(HalfCircle())
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        self.assertTrue(np.linalg.norm(np.array([0, -1.]) - last_trace) < .01)

    def test_nonlinearA(self):
        problem = MathematicalProgramTraced(
            NonlinearA(), max_evaluate=100000000)
        solver = self.Solver()
        solver.setProblem((problem))
        output = solver.solve()
        last_trace = problem.trace_x[-1]
        solution = np.array([1.00000000, 0])
        self.assertTrue(np.linalg.norm(solution - last_trace) < .01)

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
