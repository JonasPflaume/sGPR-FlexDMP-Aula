import numpy as np
from trajencoder.flexdmp.utils import RBF, integrate_RK4
from trajencoder.flexdmp.canonical_system import canonical_system, canonical_system_linear

import copy
from collections.abc import Iterable
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class ForcingFlexDMP:
    ''' <Motivation:    Higher order DMP>
        This DMP was implemented in vectorized form,
        unlike the implemented vanilla DMP
    '''
    def __init__(self, demo_len, demo_dt, p=6.5, tau=1., dof=1, linear_decay=True):
        '''
        bf_num:     The number of Psi activation points
        demo_len:   The length of the trajectory
        demo_dt:    The time interval of demonstration
        p:          The value to determine the alpha_{1,2,3}

        Marks:      The canonical system can pass in a variable has same length as demo_len to adjust the speed of state x
        '''
        if linear_decay:
            self.cs = canonical_system_linear(demo_len, demo_dt, tau=tau)
        else:
            self.cs = canonical_system(demo_len, demo_dt, tau=tau)
        self.demo_len = demo_len
        self.demo_dt = demo_dt
        self.tau = tau
        self.dof = dof
        
        self.timesteps = demo_len
        # calculate the factors in ODE
        self.alpha1, self.alpha2, self.alpha3 = 3*p, p, 1/3*p               # hand tuned the factor p
    
    def get_forcing_ground_trueth(self, y_demo, dy_demo=None, ddy_demo=None):
        '''
        The input trajectories should have shape (length, dof)
        '''
        assert y_demo.shape[1] == self.dof, "Please input the trajectory has shape (length, dof)"
        self.y10 = copy.deepcopy(y_demo[0]) # (dof)
        self.goal = copy.deepcopy(y_demo[-1]) # (dof)
        # if there is no dy ddy input
        if type(dy_demo) != np.ndarray and self.dof==1:
            dy_demo = np.gradient(y_demo.squeeze()) / self.demo_dt
            dy_demo = dy_demo[:, np.newaxis]
        elif type(dy_demo) != np.ndarray and self.dof>1:
            dy_demo = np.gradient(y_demo)[0] / self.demo_dt
        if type(ddy_demo) != np.ndarray and self.dof==1:
            ddy_demo = np.gradient(dy_demo.squeeze()) / self.demo_dt
            ddy_demo = ddy_demo[:, np.newaxis]
        elif type(ddy_demo) != np.ndarray and self.dof>1:
            ddy_demo = np.gradient(dy_demo)[0] / self.demo_dt
        
        if self.dof==1:
            dddy_demo = np.gradient(ddy_demo.squeeze()) / self.demo_dt
            dddy_demo = dddy_demo[:, np.newaxis]
        elif self.dof>1:
            dddy_demo = np.gradient(ddy_demo)[0] / self.demo_dt

        self.dy10 = copy.deepcopy(dy_demo[0]) # (dof)
        self.ddy10 = copy.deepcopy(ddy_demo[0]) # (dof)
        # f_target has shape (length, dof)
        f_target = self.tau**3 * dddy_demo + self.alpha1*self.alpha2*self.alpha3*(y_demo-self.goal[np.newaxis, :]) + \
                    self.tau * self.alpha1*self.alpha2 * dy_demo + self.tau**2 * self.alpha1 * ddy_demo
        
        self.reset_state()
        return f_target

    def reset_state(self):
        ''' reset the states for integration
        '''
        self.y = copy.deepcopy(self.y10)
        self.dy = np.zeros_like(self.dy10)
        self.ddy = np.zeros_like(self.ddy10)
        self.cs.reset_state()

    def step(self, f):
        ''' integrate the DMP system step by step
            f:
                forcing term comes from predictor
        '''

        # self.y += (1/self.tau * self.dy) * self.demo_dt
        # self.dy += (1/(self.tau)*self.ddy) * self.demo_dt
        # self.ddy += (1/self.tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - self.ddy) + f)) * self.demo_dt
        def dddy(ddy):
            return (1/self.tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - ddy) + f))
        self.ddy = integrate_RK4(self.ddy, dddy, self.demo_dt)

        def dy(y):
            return (1/self.tau * self.dy)
        self.y = integrate_RK4(self.y, dy, self.demo_dt)

        def ddy(dy):
            return (1/(self.tau)*self.ddy)
        
        self.dy = integrate_RK4(self.dy, ddy, self.demo_dt)
        return self.y.copy(), self.dy.copy(), self.ddy.copy() # all (dof,)

    def trajectory(self, predictor):
        ''' reconstruct the whole trajectory
        '''
        self.reset_state()
        timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros([timesteps, self.dof])
        dy_track = np.zeros([timesteps, self.dof])
        ddy_track = np.zeros([timesteps, self.dof])

        for t in range(timesteps):
            # run and record timestep
            if t == 0:
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.y10, self.dy10, self.ddy10
            else:
                x = np.concatenate([self.goal-self.y, self.y, self.dy, self.ddy, np.array([t], dtype=float)]) # (25,)
                x = x[np.newaxis, :]
                x = torch.from_numpy(x).float().to(device)
                f = predictor(x)
                f = f.detach().cpu().numpy().squeeze()
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.step(f)

        return y_track, dy_track, ddy_track