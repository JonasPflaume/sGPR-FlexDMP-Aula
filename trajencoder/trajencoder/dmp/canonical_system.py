import numpy as np
from trajencoder.dmp.utils import integrate_RK4

class canonical_system:
    ''' canolical system time scaling
    '''
    def __init__(self, dt, demo_len, demo_dt, ax=1.0, tau=1.0):
        self.ax = ax
        self.tau = tau
        self.dt = dt
        self.runtime = demo_len * demo_dt

        self.timesteps = int(self.runtime / self.dt)
        self.reset_state()

    def reset_state(self):
        ''' system state and time scale reset to 0
        '''
        self.x = 1.0

    def trajectory(self):
        ''' generate cs trajectory
        '''
        self.timesteps = int(self.timesteps)
        self.x_track = np.zeros(self.timesteps)
        self.reset_state()

        for t in range(self.timesteps):
            self.x_track[t] = self.x
            self.step()

        return self.x_track

    def step(self):
        ''' one step ahead simulation
        '''
        def odefunc(x):
            return (- self.ax * x) * self.tau

        self.x = integrate_RK4(self.x, odefunc, self.dt)
        return self.x
