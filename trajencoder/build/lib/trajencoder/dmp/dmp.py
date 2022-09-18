import numpy as np
from trajencoder.dmp.canonical_system import canonical_system
from trajencoder.dmp.utils import RBF, integrate_RK4

import copy
from collections.abc import Iterable

class DMP:
    def __init__(self, bf_num, demo_len, demo_dt, ay=25.0, by=6.25, dt=0.01, y0=0, goal=1, tau=1.0):
        ''' 1. Initialize DMP instantiation
            2. Use get_weight to generate the w
            3. trajectory can imitate the trajectory
        '''
        self.bf_num = bf_num
        # self.dt = dt
        self.ay = ay
        self.by = by
        self.y0 = y0
        self.goal = goal

        self.tau = tau
        self.dt = demo_dt
        self.cs = canonical_system(self.dt, demo_len, demo_dt, tau=self.tau)

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.runtime, self.bf_num)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t, here exp is the solution of the cs system!!!
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

        self.h = np.ones(self.bf_num) * self.bf_num ** 2 / self.c / self.cs.ax

        self.act_l = [RBF(center, width) for center, width in zip(self.c, self.h)]

        self.timesteps = int(self.cs.runtime / self.dt)
        self.reset_state()
        self.w = np.zeros(self.bf_num) # initialize as zeros

    def get_psi(self, x_track):
        ''' use canonical system state to generate the psi
        '''
        if isinstance(x_track, Iterable):
            psi = np.zeros([self.bf_num, len(x_track)])
            for i in range(self.bf_num):
                psi[i, :] = self.act_l[i](x_track)
        else:
            psi = np.zeros(self.bf_num)
            for i in range(self.bf_num):
                psi[i] = self.act_l[i](x_track)
        return psi

    def get_weights(self, y_demo, dy_demo=None, ddy_demo=None):
        ''' use regression to calculate the weights of basis functions
        '''
        self.y0 = y_demo[0].copy()
        self.goal = y_demo[-1].copy()
        self.y_demo = y_demo.copy()
        import scipy.interpolate
        path = np.zeros(self.timesteps)
        x = np.linspace(0, self.cs.runtime, len(y_demo))

        path_gen = scipy.interpolate.interp1d(x, y_demo)
        for t in range(self.timesteps):
            path[t] = path_gen(t * self.dt)
        
        y_demo = path
        self.y_demo = y_demo.copy() # use to plot

        if dy_demo != np.ndarray:
            dy_demo = np.gradient(y_demo) / self.dt
        if ddy_demo != np.ndarray:
            ddy_demo = np.gradient(dy_demo) / self.dt
        
        f_target = np.zeros(y_demo.shape[0])

        f_target = ddy_demo - self.ay * (self.by * (self.goal - y_demo) - dy_demo)
        # start the weighted linear regression

        k = self.goal - self.y0
        x_track = self.cs.trajectory()
        self.w = np.zeros(self.bf_num)

        psi = self.get_psi(x_track)
        for b in range(self.bf_num):
            num = np.sum(x_track * psi[b, :] * f_target)
            den = np.sum(x_track ** 2 * psi[b, :])

            self.w[b] = num / den
            # if abs(k) > 1e-2:
            #     self.w[b] /= k
        self.w = np.nan_to_num(self.w)
        return self.w

    def set_weight(self, w, y0, goal):
        ''' set the existing DMP weights, init and end position
        '''
        self.y0 = y0
        self.goal = goal
        self.w = w

    def reset_state(self):
        ''' reset the state including the canonical system
        '''
        self.y = copy.deepcopy(self.y0)
        self.dy = 0.0
        self.ddy = 0.0
        self.cs.reset_state()

    def step(self):
        ''' one step ahead simulation of transformation system
        '''
        # forcing term
        x = self.cs.step()
        psi = self.get_psi(x)

        # remove the last term to tackle the collapse when very near start and end points were given
        front_term = x #* (self.goal - self.y0)
        f = front_term * np.dot(psi, self.w)
        sum_psi = np.sum(psi)

        if np.abs(sum_psi) > 1e-6:
            f /= sum_psi

        # acceleration
        self.ddy = (
                self.ay * (self.by * (self.goal - self.y) - self.dy) + f
            )
        def ddy_ode(dy):
            return self.tau * self.ay * (self.by * (self.goal - self.y) - dy) + f

        self.dy = integrate_RK4(self.dy, ddy_ode, self.dt)

        def dy_ode(dy):
            # dy for placeholder, not used
            return self.tau * self.dy
        self.y  = integrate_RK4(self.y, dy_ode, self.dt)

        return self.y, self.dy, self.ddy
    
    def trajectory(self, y0=None, goal=None, tau=None):
        """Generate a system trial, no feedback is incorporated."""
        if y0 != None:
            self.y0 = y0
        if goal != None:
            self.goal = goal
        if tau != None:
            self.tau = tau

        self.reset_state()
        timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros(timesteps)
        dy_track = np.zeros(timesteps)
        ddy_track = np.zeros(timesteps)

        for t in range(timesteps):
            # run and record timestep
            y_track[t], dy_track[t], ddy_track[t] = self.step()

        return y_track, dy_track, ddy_track
