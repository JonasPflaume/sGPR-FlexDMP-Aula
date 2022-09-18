import numpy as np
from trajencoder.flexdmp.utils import RBF, integrate_RK4
import copy

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
        self.demo_dt = demo_dt
        self.runtime = demo_len * demo_dt
        self.timesteps = demo_len
        self.time = np.linspace(0, demo_dt*demo_len, demo_len)
        if tau is not None:
            self.tau = tau
        else:
            self.tau = 1.

    def reset_state(self):
        ''' system state reset/initialize to 1.
        '''
        self.x = 1.0

    def trajectory(self):
        ''' generate cs trajectory
        '''
        x_track = []
        self.reset_state()
        for t in range(1000000):
            x_track.append( copy.deepcopy(self.x) )
            self.x += - 1/(self.tau*self.runtime) * self.demo_dt
            if self.x <= 0:
                break
        self.reset_state()
        return np.array( x_track )

class PhaseFreezeFlexDMP:
    ''' <Motivation:    Higher order DMP>
        This DMP was implemented in vectorized form,
        unlike the implemented vanilla DMP, with phase freeze function
        the canonical system is linear decay, in this case!
    '''
    def __init__(self, bf_num, demo_len, demo_dt, p=11.0, w_factor=2.5, dof=1, default_tau=1.0) -> None:
        self.bf_num = bf_num
        self.demo_len = demo_len
        self.demo_dt = demo_dt
        self.w_factor = w_factor
        self.dof = dof
        
        self.default_tau = default_tau # for weights calculation
        self.c = np.linspace(default_tau, 0, self.bf_num)
        sigma = np.abs(self.c[1]-self.c[0])/2 * np.sqrt(-1/(2*np.log(0.5))) * np.ones(self.bf_num)
        self.h = 1 / (2*sigma**2)
        self.h *= w_factor
        self.rbf_l = [RBF(center, width) for center, width in zip(self.c, self.h)]
        self.cs = canonical_system_linear(demo_len, demo_dt, tau=default_tau)
        
        self.alpha1, self.alpha2, self.alpha3 = 3*p, p, 1/3*p
        
    def __repr__(self) -> str:
        return f"<PhaseFreezeFlexDMP with {self.bf_num} basis for {self.dof} dof.>"
    
    def get_weights(self, y_demo, dy_demo=None, ddy_demo=None):
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

        self.dy10 = copy.deepcopy(dy_demo[0]) # (dof,)
        self.ddy10 = copy.deepcopy(ddy_demo[0]) # (dof,)
        # f_target has shape (length, dof)
        f_target = self.default_tau**3 * dddy_demo + self.alpha1*self.alpha2*self.alpha3*(y_demo-self.goal[np.newaxis, :]) + \
                    self.default_tau * self.alpha1*self.alpha2 * dy_demo + self.default_tau**2 * self.alpha1 * ddy_demo
        self.weights = np.zeros([self.dof, self.bf_num])
        x = self.cs.trajectory() # shape (length,)
        s = x[:, np.newaxis]# * ((self.goal - self.y10)[np.newaxis, :]) # shape (length, dof)s
        for i in range(self.bf_num):
            Gamma_i = self.rbf_l[i](x)
            w_i = (s * Gamma_i[:, np.newaxis] * f_target).sum(axis=0) / (s**2 * Gamma_i[:, np.newaxis]).sum(axis=0) # shape (dof,)
            # print(w_i)
            self.weights[:,i] = w_i
        self.weights = np.nan_to_num(self.weights)
        self.reset_state()
        return self.weights

    def set_weight(self, weights, initial, goal):
        ''' the weights should have the shape (dof, bf_num)
            in the same time you have to give initial and goal of traj with shape (dof,)
        '''
        assert type(weights) == np.ndarray and weights.shape == (self.dof, self.bf_num), "Use the right weight format please!"
        assert type(initial) == np.ndarray and initial.shape == (self.dof,), "Use the right initial format please!"
        assert type(goal) == np.ndarray and goal.shape == (self.dof,), "Use the right goal format please!"
        self.weights = weights
        self.y10 = copy.deepcopy(initial)
        self.dy10 = np.zeros(self.dof)
        self.ddy10 = np.zeros(self.dof)
        self.goal = goal
        
        
    ####################################### pure dynamical system simulation ! ###################################
    def reset_state(self):
        self.y = copy.deepcopy(self.y10)
        self.dy = np.zeros_like(self.dy10)
        self.ddy = np.zeros_like(self.ddy10)
        self.cs.reset_state()
        self.phase = 1.0
        
    def step(self, phase_factor, violation):
        ''' integrate the DMP system step by step
        '''
        # forcing term
        psi = np.array([self.rbf_l[i](self.phase) for i in range(self.bf_num)]) # (bf_num,)

        # remove the last term to tackle the collapse when very near start and end points were given
        front_term = self.phase# * (self.goal - self.y10) # (dof,)
        f = front_term * np.einsum('ij,j->i', self.weights, psi) # (dof, num) x (num) -> (dof,)

        sum_psi = np.sum(psi)
        if np.abs(sum_psi) > 1e-8:
            f /= sum_psi
        
        # change the length
        phase_tau = 1.55 * self.default_tau
        #
        
        def dddy(ddy):
            return (1/phase_tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal - self.y) - self.dy) - self.ddy) + f))
        self.ddy = integrate_RK4(self.ddy, dddy, self.demo_dt)
        
        def ddy(dy):
            return (1/phase_tau * self.ddy)
        self.dy = integrate_RK4(self.dy, ddy, self.demo_dt)

        def dy(y):
            return (1/phase_tau * self.dy)
        self.y = integrate_RK4(self.y, dy, self.demo_dt)

        self.phase += 1/(phase_tau) * phase_factor * self.demo_dt
        
        return self.y, 1 / phase_tau * self.dy, 1 / phase_tau ** 2 * self.ddy # all (dof,)

    def trajectory(self, tau_func, torque_limits):
        ''' reconstruct the whole trajectory
        '''
        self.reset_state()
        # set up tracking vectors
        y_track = []
        dy_track = []
        ddy_track = []

        for t in range(1000000):
            # run and record timestep
            torque = tau_func(self.y, self.dy, self.ddy).full().squeeze()
            violation = np.abs( torque ) - torque_limits
            violation[violation < 0.] = 0.
            violation = np.max(violation)

            phase_factor = - 1/(self.default_tau * (self.demo_dt * self.demo_len) ) ### increase the length ... 
            if t == 0:
                y_track.append(self.y10.reshape(1,-1))
                dy_track.append(self.dy10.reshape(1,-1))
                ddy_track.append(self.ddy10.reshape(1,-1))
            else:
                y_t, dy_t, ddy_t = self.step(phase_factor, violation) # and here...
                y_track.append(y_t.copy().reshape(1,-1))
                dy_track.append(dy_t.copy().reshape(1,-1))
                ddy_track.append(ddy_t.copy().reshape(1,-1))
            if self.phase <= 0:
                break
        y_track, dy_track, ddy_track = np.concatenate(y_track), np.concatenate(dy_track), np.concatenate(ddy_track)
        return y_track, dy_track, ddy_track
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    ## test run ##
    show_plot = True
    import h5py
    import random
    from casadi_kinodynamics.utils import symbolic_robot
    sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                          tool_mass=0.,
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.0],
                                          tool_inertia=[
                                              0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)
    tau_func = sym_robot.inv_dyn
    tau_max = sym_robot.tau_max
                
    FILE = h5py.File('/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_1e-07_jerk_2.5_mass.hdf5')
    random_index = list(FILE.keys())[1] # 129
    y_demo, dy_demo, ddy_demo = np.array(FILE[random_index]['q']), np.array(FILE[random_index]['qd']), np.array(FILE[random_index]['qdd'])
    length = len(y_demo)
    dt = 0.004 
    t_demo = np.linspace(0., len(y_demo)*dt, len(y_demo))
    # DMP
    dmp_demo = PhaseFreezeFlexDMP(50, demo_len=length, demo_dt=dt, dof=6)  # order=30 is the elbow of Loss curve
    dmp_demo.get_weights(y_demo, dy_demo=dy_demo, ddy_demo=ddy_demo)
    y, dy, ddy = dmp_demo.trajectory(tau_func, tau_max) # 0.01 * 200 ->  2s
    
    tau = []
    for j in range(len(y)):
        tau.append(sym_robot.inv_dyn(y[j], dy[j], ddy[j]).full().reshape(1,-1))
    tau = np.concatenate(tau)
    violation = np.abs(tau) - np.array(sym_robot.tau_max)
    violation[violation<0.] = 0.
    print(100 * violation.max(axis=0) / np.array(sym_robot.tau_max))

    from scipy.interpolate import CubicSpline
    slow_factor = 0
    t = np.linspace(0., len(y)*dt*(1+slow_factor/100), len(y))
    t_normal = np.linspace(0., len(y)*dt, len(y))
    cs = CubicSpline(t, dy*(1/(1+slow_factor/100)))
    y_cs = []
    y0_cs = y[0].copy()
    for tt in t:
        y_cs.append(y0_cs.copy().reshape(1,-1))
        y0_cs += cs(tt) * (1+slow_factor/100) * dt
        
    y_cs = np.concatenate(y_cs)
    dy_cs = cs(t)
    ddy_cs = cs(t, 1)
    
    tau = []
    for j in range(len(y_cs)):
        tau.append(sym_robot.inv_dyn(y_cs[j], dy_cs[j], ddy_cs[j]).full().reshape(1,-1))
    tau = np.concatenate(tau)
    violation = np.abs(tau) - np.array(sym_robot.tau_max)
    violation[violation<0.] = 0.
    print(100 * violation.max(axis=0) / np.array(sym_robot.tau_max))
    
    if show_plot:
        plt.figure(figsize=[5,7])
        
        plt.subplot(3,1,1)
        plt.plot(t_demo, y_demo, '-.r', label="demo")
        plt.plot(t_normal, y, '-b', label="dmp")
        plt.plot(t, y_cs, '-c', alpha=0.5, label="it")
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc='lower right')
        plt.grid()

        plt.subplot(3,1,2)
        plt.plot(t_demo, dy_demo, '-.r')
        plt.plot(t_normal, dy, '-b')
        plt.plot(t, cs(t), '-c', alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # plt.legend(handles, labels, loc='best')
        plt.grid()
        # plt.savefig("../plots/traj_encoding/DMP_BSpline.jpg", dpi=200)

        plt.subplot(3,1,3)
        plt.plot(t_demo, ddy_demo, '-.r')
        plt.plot(t_normal, ddy, '-b')
        plt.plot(t, cs(t, 1), '-c', alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        # plt.legend(handles, labels, loc='best')
        plt.grid()
        plt.savefig("/home/jiayun/Desktop/dmp.jpg", dpi=200)
        plt.show()