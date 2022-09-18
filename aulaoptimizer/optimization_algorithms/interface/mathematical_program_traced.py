from .mathematical_program import MathematicalProgram


class MathematicalProgramTraced(MathematicalProgram):

    def __init__(self, mathematical_program, max_evaluate=1000):
        self.mathematical_program = mathematical_program

        self.counter_evaluate = 0
        self.counter_hessian = 0
        self.max_evaluate = max_evaluate

        self.trace_x = []
        self.trace_phi = []
        self.trace_J = []

        self.trace_x_hessian = []

        super().__init__()

    def reset_counters(self):
        """
        """
        self.counter_evaluate = 0
        self.counter_hessian = 0

        self.trace_x = []
        self.trace_phi = []
        self.trace_J = []

        self.trace_x_hessian = []

    def evaluate(self, x):
        """
        """
        if self.counter_evaluate > self.max_evaluate:
            raise RuntimeError("too many iterations, returning")
        self.counter_evaluate += 1
        phi, J = self.mathematical_program.evaluate(x)
        self.appendToTrace(x, phi, J)
        return phi, J

    def appendToTrace(self, x, phi, J):
        """
        This should be called at the end of an evaluate implementation. It adds
        the evaluated x, features and objectives to the traces
        """
        self.trace_x.append(x.copy())
        self.trace_J.append(J.copy())
        self.trace_phi.append(phi.copy())

    def getBounds(self):
        """
        """
        return self.mathematical_program.getBounds()

    def getDimension(self):
        """
        """
        return self.mathematical_program.getDimension()

    def getFHessian(self, x):
        """
        """
        self.counter_hessian += 1
        self.trace_x_hessian.append(x)
        return self.mathematical_program.getFHessian(x)

    def getFeatureTypes(self):
        """
        """
        return self.mathematical_program.getFeatureTypes()

    def getInitializationSample(self):
        """
        """
        return self.mathematical_program.getInitializationSample()

    def report(self, verbose):
        """
        """
        header = "Traced Mathematical Program\n"
        out = self.mathematical_program.report(verbose)
        return header + out
