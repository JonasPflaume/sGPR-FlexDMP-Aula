import numpy as np
import sys, os

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT

class SolverAugmentedLagrangian(NLPSolver):
    ''' Augmented lagrangian optimizer, implemented follows
        "A Novel Augmented Lagrangian Approach for
        Inequalities and Convergent Any-Time Non-Central
        Updates" -- Marc Toussaint 2014
    '''

    def __init__(self):
        self.theta = 1e-5 # x tolerance
        self.epsi = 1e-4  # constraints tolerance
        self.verbose = 0  # print intermediate info?

    @property
    def curr_q(self):
        ''' count how many times the defined opt. problem are visiting
            a metric to evalute the efficiency of algorithm
        '''
        return self.problem.counter_evaluate

    def solve(self):
        ''' where the augmented lagrangian main loop defined 
        '''
        FT = self.problem.getFeatureTypes()
        self.obj_idx = [i for i, item in enumerate(FT) if item == 1]
        self.sos_idx = [i for i, item in enumerate(FT) if item == 2]
        self.inq_idx = [i for i, item in enumerate(FT) if item == 3]
        self.eq_idx = [i for i, item in enumerate(FT) if item == 4]
        # initialize the parameters
        rho_mu = rho_v = 3.
        mu = v = 1.
        # solveUC will handle the empty list
        labd = np.zeros(len(self.inq_idx))
        kappa = np.zeros(len(self.eq_idx))
        x0 = self.problem.getInitializationSample()
        x0 = x0.reshape(-1,1)
        for i in range(1000):
            if self.verbose:
                print("___%i iteration__ curr_q: %i" % (i, self.curr_q))
            # 1. solve the unconstrained optimization
            x1 = self.solveUC(x0, mu, labd, v, kappa)
            # 2. update the parameters
            phi, J = self.problem.evaluate(x1.squeeze())
            if self.inq_idx:
                g_x = phi[self.inq_idx]
                labd = np.maximum(labd + 2 * mu * g_x, np.zeros_like(labd))
            if self.eq_idx:
                h_x = phi[self.eq_idx]
                kappa += 2 * v * h_x
            # 3. * try to update the mu and v (LP:this will increase the complexity in inner loop)
            mu *= rho_mu
            v *= rho_v
            # 4. check the stop criterion by enumeration
            if self.inq_idx and not self.eq_idx:
                if np.linalg.norm(x1-x0) < self.theta and np.all(g_x < self.epsi):
                    break
            elif self.eq_idx and not self.inq_idx:
                if np.linalg.norm(x1-x0) < self.theta and np.all(np.abs(h_x) < self.epsi):
                    break
            elif not self.eq_idx and not self.inq_idx:
                if np.linalg.norm(x1-x0) < self.theta:
                    break
            elif self.eq_idx and self.inq_idx:
                if np.linalg.norm(x1-x0) < self.theta and np.all(g_x < self.epsi) and np.all(np.abs(h_x) < self.epsi):
                    break
            if self.curr_q > 20000:
                print('Exceed the maximun query time...')
                break
            x0 = x1
        print("Query time: %i" % self.curr_q)
        return x1
        
    def solveUC(self, x0, mu, labd, v, kappa):
        ''' solve the unconstrained inner loop
            input:
                    x0:     the current x value
                    mu:     the current Aula parameters for square term of inequality constraints
                    labd:   the current Aula parameters for augmented lagrangian multiplyer of ineq
                    v:      the current Aula parameters for square term of equality constraints
                    kappa:  the current Aula parameters for augmented lagrangian multiplyer of eq
            output:
                    x1:     the solution of inner loop
        '''
        tolerance = 1e-4
        damping = 0
        verbose = self.verbose
        damping_decrease = 0.1

        rho_a_in = 1.2
        rho_a_de = 0.5
        sigmax = 10
        rho_ls = 0.01
        lr = 1

        def BTincrease(lr):
            return np.min([lr * rho_a_in, sigmax])

        def BTdecrease(lr, curr_x, direction):
            while True:
                lhs_x = curr_x + lr * direction
                c, J, _= self.AulaEvaluate(lhs_x, mu, labd, v, kappa)
                lhs = c
                rhs, J, _= self.AulaEvaluate(curr_x, mu, labd, v, kappa)
                grad = J.reshape(-1,1)
                direction = direction.reshape(-1,1)
                rhs += rho_ls * grad.T @ (lr * direction)
                if lr < 1e-20:
                    print("WARNING: Can't find a proper lr, breaking...")
                    break
                if lhs > rhs or np.isnan(lhs):
                    lr *= rho_a_de
                else:
                    break
            return lr

        def get_direction(J, H):
            grad = J.reshape(-1,1)
            try:
                direction = -np.linalg.inv(H) @ grad
            except:
                direction = -grad

            PULLBACK = False
            if grad.T @ direction > 0:
                if self.verbose:
                    print('---PULLBACK---')
                direction = -grad
                PULLBACK = True
            return direction, PULLBACK

        x0 = x0.reshape(-1,1)
        PB_counter = 0
        for i in range(1500):
            if self.verbose:
                print('step {}, x = {}'.format(i, x0))
            # 1. get the update direction
            phi, J, H = self.AulaEvaluate(x0, mu, labd, v, kappa)
            direction, PB = get_direction(J, H)
            # break loop if too many pullback
            if bool(PB):
                PB_counter += int(PB)
            else:
                PB_counter = 0
            # 2. run the BT linear search
            lr = BTdecrease(lr, x0, direction)
            
            # 4. break loop
            if np.linalg.norm(lr*direction) < tolerance or PB_counter > 5:
                x1 = x0
                break
            # 3. update x
            x1 = x0 + lr * direction
            
            x0 = x1
            lr = BTincrease(lr)
        return x1

    def AulaEvaluate(self, x, mu, labd, v, kappa):
        ''' wrap the defined problem to get the aula needed phi, gradient and hessian
            input:
                    x0:     the current x value
                    mu:     the current Aula parameters for square term of inequality constraints
                    labd:   the current Aula parameters for augmented lagrangian multiplyer of ineq
                    v:      the current Aula parameters for square term of equality constraints
                    kappa:  the current Aula parameters for augmented lagrangian multiplyer of eq
            output:
                    phi:    feature vector for aula inner loop
                    J:      Jaconbian for aula inner loop
                    H:      Hessian matrix for aula inner loop
        '''
        labd = labd.reshape(-1,1)
        kappa = kappa.reshape(-1,1)
        # return the uc phi, J and H
        obj_idx = self.obj_idx
        sos_idx = self.sos_idx
        inq_idx = self.inq_idx
        eq_idx = self.eq_idx

        phi = np.zeros([1,1])
        J = np.zeros([1, self.problem.getDimension()])
        H = np.zeros([self.problem.getDimension(), self.problem.getDimension()])

        phi0, J0 = self.problem.evaluate(x.squeeze())
        try:
            H_f0 = self.problem.getFHessian(x.squeeze())
        except:
            H_f0 = 0
        # get the necessary value, grad and ggrad
        # 1. get the phi_obj
        # 2. compute the J
        # 3. compute the H
        if inq_idx:
            I_labd = np.logical_or(phi0[inq_idx] >= 0, labd.squeeze() > 0)
            I_labd = np.array(I_labd, dtype=float)
            I_labd = np.diag( I_labd )
            g_x = phi0[inq_idx].reshape(-1,1)
            grad_g_x = J0[inq_idx]
            inq_term = ((mu * I_labd @ g_x + labd).T @ g_x).astype(float)
            phi += inq_term
            J += (2. * mu * I_labd @ g_x + labd).T @ grad_g_x
            H += 2. * mu * grad_g_x.T @ I_labd @ grad_g_x
        if eq_idx:
            h_x = phi0[eq_idx].reshape(-1,1)
            grad_h_x = J0[eq_idx]
            eq_term = (v * h_x + kappa).T @ h_x
            phi += eq_term
            J += (2 * v * h_x + kappa).T @ grad_h_x
            H += 2 * v * grad_h_x.T @ grad_h_x
        if obj_idx:
            grad_f_x = J0[obj_idx].reshape(1,-1)
            obj_term = np.sum(phi0[obj_idx])
            phi += obj_term
            J += grad_f_x
            H += H_f0
        if sos_idx:
            sos_term = np.inner(phi0[sos_idx], phi0[sos_idx])
            phi += sos_term
            Fsos = phi0[sos_idx].reshape(-1,1)
            DxDFsos = J0[sos_idx].reshape(len(sos_idx), self.problem.getDimension())
            J += 2 * (Fsos.T @ DxDFsos)
            H += 2 * DxDFsos.T @ DxDFsos
        assert len(phi) == 1
        assert J.shape[0] == 1
        return phi, J, H