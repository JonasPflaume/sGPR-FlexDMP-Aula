from .mathematical_program import MathematicalProgram


class NLPSolver():
    """
    A Non linear Solver that solves a Nonlinear Program:

    min_x  f(x) + phi_sos(x)^T phi_sos(x)
    s.t.     phi_eq(x) = 0
             phi_ineq(x) <= 0
             B_lo <= x <= B_up

    The NLP should be implemented as an object of the class MathematicalProgram

    See Also
    -----
    MathematicalProgram

    """

    def __init__(self):
        """
        """
        self.problem = MathematicalProgram()

    def setProblem(self, mathematical_program):
        """

        Arguments
        ----
        mathematicalProgram: MathematicalProgram

        """
        self.problem = mathematical_program

    def solve(self):
        """
        Solve current nonlinear program. Returns the convergence point x 
       (1-D np.array)

        Returns:
        x: np-array 1-D 

        Notes
        ----

        use:
        x = self.problem.getInitializationSample()
        to get the starting point of the algorithm

        use:
        phi , J = self.problem.evaluate(x)
        to evaluate the Mathematical program at a variable value x.


        See Also:
        ----
        MathematicalProgram.getInitializationSample
        MathematicalProgram.evaluate

        """
        raise NotImplementedError()
