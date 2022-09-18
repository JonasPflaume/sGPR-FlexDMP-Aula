import numpy as np
from .objective_type import OT


class MathematicalProgram():
    """
    Non Linear program

    min_x  f(x) + phi_sos(x)^T phi_sos(x)
    s.t.     phi_eq(x) = 0
             phi_ineq(x) <= 0
             B_lo <= x <= B_up

    where: 
    x is a continous variable, in vector space R^n
    f is a scalar function
    phi_sos is a vector of residuals (sum-of-square cost term)
    phi_eq is a vector of equality constraints
    phi_ineq is a vector of inequality constraints
    B_lo and B_up are, respectively, the lower and upper bounds


    See Also:
    -----
    MathematicalProgramTraced

    """

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, x):
        """
        query the NLP at a point x; returns the tuple (phi,J), which is
        the feature vector and its Jacobian; features define cost terms, 
        sum-of-square (sos) terms, inequalities, and 
        equalities depending on 'getFeatureTypes'

        Parameters
        ------
        x: np.array, 1-D

        Returns
        ------
        phi: np.array 1-D
        J: np.array 2-D.  J[i,j] is derivative of feature i w.r.t variable j

        """
        raise NotImplementedError()

    def getCost(self, x):
        """
        cost terms at point x; i.e.
        f(x) + phi_sos(x)^T phi_sos(x)

        Returns:
        ----
        output: float
        """

        types = self.getFeatureTypes()
        index_f = [i for i, x in enumerate(types) if x == OT.f]
        assert(len(index_f) <= 1)  # at most, only one term of type OT.f
        index_r = [i for i, x in enumerate(types) if x == OT.sos]

        def f(x):
            phi, _ = self.evaluate(x)
            c = 0  # cost
            if len(index_f) > 0:
                c += phi[index_f][0]
            if len(index_r) > 0:
                c += phi[index_r].T @ phi[index_r]
            return c

        return f(x)

    def getDimension(self):
        """
        return the dimensionality of x

        Returns
        -----
        output: integer

        """
        raise NotImplementedError()

    def getBounds(self):
        """
        returns the tuple (b_lo,b_up), where both vectors are of same dimensionality of x (or size zero, if there are no bounds)

        Returns
        ------
        b_lo: np.array 1D
        b_up: np.array 1D

        """
        n = self.getDimension()
        return (np.repeat(-np.Inf, n), np.repeat(+np.Inf, n))

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]

    def getFHessian(self, x):
        """
        returns Hessian of the sum of $f$-objectives

        Returns
        -----
        hessian: np.array 2D

        """
        raise NotImplementedError()

    def getInitializationSample(self):
        """
        returns a sample (e.g. uniform within bounds) to initialize an optimization -- not necessarily feasible

        Returns
        -----
        x:  np.array 1-D

        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        displays semantic information on the last query

        Parameters
        ----

        Returns
        ----
        output: string
        """
        raise NotImplementedError()
