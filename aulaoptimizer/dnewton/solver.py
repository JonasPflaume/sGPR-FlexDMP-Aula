import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT

class SolverUnconstrained(NLPSolver):

    def __init__(self):
        self.tolerance = 1e-5
        self.damping = 0
        self.verbose = 1
        self.damping_decrease = 0.1
        self.use_backtracking = 1

    def solve(self):
        backtracking = BackTracking(self.problem)

        x0 = self.problem.getInitializationSample() # (s,)
        x0 = x0.reshape(-1,1)

        c, Jacobian, H = getCJH(self.problem, x0)

        lr = 1
        for epi in range(10000):
            gradient = Jacobian.reshape(-1,1)
            direction = self.get_direction(gradient, H)
            if self.use_backtracking:
                lr = backtracking.decrease(lr, x0, direction)

            x1 = x0 + lr * direction
            c, Jacobian, H = getCJH(self.problem, x1)
            if self.verbose:
                print("At step %i || current cost: %.8f" % (epi, c))

            cron = np.linalg.norm(Jacobian) < self.tolerance
            cron1 = np.linalg.norm(lr*direction) < (self.tolerance * 1e-4)
            if cron or cron1:
                break
            x0 = x1
            if self.use_backtracking:
                lr = backtracking.increase(lr)
            self.damping *= self.damping_decrease
        if self.verbose:
            print('Solver ended...')
        return x1

    def get_direction(self, gradient, hessian):
        if type(hessian) == np.ndarray:
            try:
                direction =  -np.linalg.inv(hessian + np.eye(self.problem.getDimension())*self.damping) @ gradient
            except:
                direction = -gradient

            while gradient.T @ direction > 0:
                self.damping = -np.min(np.linalg.eig(hessian)[0]) + 0.7 # greater than min eig value
                direction = -np.linalg.inv(hessian + np.eye(self.problem.getDimension())*self.damping) @ gradient
        else:
            direction = -gradient
        return direction


class BackTracking(object):
    def __init__(self, problem):
        self.rho_a_in = 1.2
        self.rho_a_de = 0.5
        self.sigmax = 10
        self.rho_ls = 0.01
        self.problem = problem

    def decrease(self, lr, curr_x, direction):
        while True:
            lhs_x = curr_x + lr * direction
            c, J, _ = getCJH(self.problem, lhs_x)
            lhs = c
            rhs, J, _ = getCJH(self.problem, curr_x)
            grad = J.reshape(-1,1)
            direction = direction.reshape(-1,1)
            rhs += (self.rho_ls * grad.T @ (lr * direction)).reshape(rhs.shape)
            if lr < 1e-12:
                raise ValueError('Fail to find a proper lr...')
            if lhs > rhs or np.isnan(lhs):
                lr *= self.rho_a_de
            else:
                break
        return lr

    def increase(self, lr):
        return np.min([lr * self.rho_a_in, self.sigmax])
            

def getCJH(problem, x):
        # helper function to get the CJH value
        types = problem.getFeatureTypes()
        index_f = [i for i, item in enumerate(types) if item == OT.f]
        index_r = [i for i, item in enumerate(types) if item == OT.sos]
        x = x.squeeze()
        phi, J = problem.evaluate(x)
        try:
            H = problem.getFHessian(x)  # if necessary
        except:
            H = 0

        # calculate the Jacobian and hessian in different situation
        Jacobian = np.zeros([1, problem.getDimension()])
        if len(index_f) > 0:
            Jacobian += J[index_f]
            if len(index_r) > 0:
                Jacobian += 2 * J[index_r].T @ phi[index_r]
                H += 2 * J[index_r].T @ J[index_r]
        else:
            if len(index_r) > 0:
                H += 2 * J[index_r].T @ J[index_r]
                Jacobian += 2 * J[index_r].T @ phi[index_r]

        c = 0
        if len(index_f) > 0:
            c += phi[index_f][0]
        if len(index_r) > 0:
            c += phi[index_r].T @ phi[index_r]

        return c, Jacobian, H # CJH (cost, Jacobian, Hessian)
