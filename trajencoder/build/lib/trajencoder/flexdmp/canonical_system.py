import numpy as np
from trajencoder.flexdmp.utils import integrate_RK4

class canonical_system:
    ''' canolical system time scaling
    '''
    def __init__(self, demo_len, demo_dt, ax=1, dt=None, tau=None):
        ''' demo_len:       The length of demo traj
            demo_dt:        The integration interval of demo traj
            dt(None):       The interval of reconstruction
            ax:             See equation DMP
            tau:            See equation DMP, here tau is designed to be vector value
                            in order to left API for adjust the time scale factor
        '''
        self.ax = ax
        self.dt = dt if dt is not None else demo_dt
        self.runtime = demo_len * demo_dt
        self.timesteps = int(self.runtime / self.dt)
        if tau is not None:
            if type(tau) == float:
                self.tau = np.ones(self.timesteps)*tau
            elif type(tau) == np.ndarray:
                self.tau = tau
        else:
            self.tau = np.ones(self.timesteps)
        self.reset_state()

    def reset_state(self):
        ''' system state reset/initialize to 1.
        '''
        self.x = 1.0
        self.counter = 0 # the counter for indexing the time factor tau

    def trajectory(self):
        ''' generate cs trajectory
        '''
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
            return (- self.ax * x) * self.tau[self.counter] # tau = 1/tau in DMP equation

        self.x = integrate_RK4(self.x, odefunc, self.dt)
        self.counter += 1
        return self.x

class canonical_system_linear:
    ''' canolical system linear decay
    '''
    def __init__(self, demo_len, demo_dt, ax=1, dt=None, tau=None):
        ''' demo_len:       The length of demo traj
            demo_dt:        The integration interval of demo traj
            dt(None):       The interval of reconstruction
            ax:             See equation DMP
            tau:            See equation DMP, here tau is designed to be vector value
                            in order to left API for adjust the time scale factor
        '''
        self.ax = ax
        self.dt = dt if dt is not None else demo_dt
        self.runtime = demo_len * demo_dt
        self.timesteps = demo_len
        self.time = np.linspace(0, demo_dt*demo_len, demo_len)
        if tau is not None:
            if type(tau) == float:
                self.tau = np.ones(self.timesteps)*tau
            elif type(tau) == np.ndarray:
                self.tau = tau
        else:
            self.tau = np.ones(self.timesteps)
        self.reset_state()

    def reset_state(self):
        ''' system state reset/initialize to 1.
        '''
        self.x = 1.0
        self.counter = 0 # the counter for indexing the time factor tau

    def trajectory(self):
        ''' generate cs trajectory
        '''
        self.x_track = np.zeros(self.timesteps)
        self.reset_state()
        for t in range(self.timesteps):
            self.x_track[t] = 1 - 1/(self.tau[t]*self.runtime) * self.time[t]
        self.reset_state()
        return self.x_track

    def step(self):
        ''' one step ahead simulation
        '''
        self.x = 1 - 1/(self.tau[self.counter]*self.runtime) * self.time[self.counter]
        self.counter += 1
        return self.x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cs = canonical_system_linear(60, 0.01)
    x_track = cs.trajectory()
    plt.plot(x_track)
    plt.show()
