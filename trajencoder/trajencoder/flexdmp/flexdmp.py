import numpy as np
from trajencoder.flexdmp.utils import RBF, integrate_RK4
from trajencoder.flexdmp.canonical_system import canonical_system, canonical_system_linear

import copy
from collections.abc import Iterable

class FlexDMP:
    ''' <Motivation:    Higher order DMP>
        This DMP was implemented in vectorized form,
        unlike the implemented vanilla DMP
    '''
    def __init__(self, bf_num, demo_len, demo_dt, p=11., w_factor=2.5, tau=1., dof=1, linear_decay=True): # p=23, w = 1.3 better for peak violation!
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
        self.bf_num = bf_num
        self.demo_len = demo_len
        self.demo_dt = demo_dt
        self.tau = tau
        self.dof = dof
        # desired activations throughout time
        if type(self.cs) == canonical_system:
            des_c = np.linspace(0, self.cs.runtime, self.bf_num)
            self.c = np.ones(self.bf_num)
            for n in range(self.bf_num):
                # finding x for desired times t, here exp is the solution of the cs system
                self.c[n] = np.exp(-self.cs.ax * des_c[n])

            self.h = w_factor*np.ones(self.bf_num) * self.bf_num ** 2 / self.c / self.cs.ax  # width increase 20 times (care for the local!)
            self.rbf_l = [RBF(center, width) for center, width in zip(self.c, self.h)]
        elif type(self.cs) == canonical_system_linear:
            self.c = np.linspace(1, 0, self.bf_num)
            sigma = np.abs(self.c[1]-self.c[0])/2 * np.sqrt(-1/(2*np.log(0.5))) * np.ones(self.bf_num)
            self.h = 1 / (2*sigma**2)
            self.h *= w_factor
            self.rbf_l = [RBF(center, width) for center, width in zip(self.c, self.h)]
        
        self.timesteps = demo_len
        # calculate the factors in ODE
        self.alpha1, self.alpha2, self.alpha3 = 3*p, p, 1/3*p               # hand tuned the factor p
    
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

        self.dy10 = copy.deepcopy(dy_demo[0]) # (dof)
        self.ddy10 = copy.deepcopy(ddy_demo[0]) # (dof)
        # f_target has shape (length, dof)
        f_target = self.tau**3 * dddy_demo + self.alpha1*self.alpha2*self.alpha3*(y_demo-self.goal[np.newaxis, :]) + \
                    self.tau * self.alpha1*self.alpha2 * dy_demo + self.tau**2 * self.alpha1 * ddy_demo
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

    def reset_state(self):
        ''' reset the states for integration
        '''
        self.y = copy.deepcopy(self.y10)
        self.dy = np.zeros_like(self.dy10)
        self.ddy = np.zeros_like(self.ddy10)
        self.cs.reset_state()

    def step(self, step_num, timesteps, slow_percent):
        ''' integrate the DMP system step by step
        '''
        # forcing term
        
        x = self.cs.step(step_num, timesteps)

        psi = np.array([self.rbf_l[i](x) for i in range(self.bf_num)]) # (bf_num,)

        # remove the last term to tackle the collapse when very near start and end points were given
        front_term = x# * (self.goal - self.y10) # (dof,)
        f = front_term * np.einsum('ij,j->i', self.weights, psi) # (dof, num) x (num) -> (dof,)
        sum_psi = np.sum(psi)
        if np.abs(sum_psi) > 1e-6:
            f /= sum_psi

        tau = (1 + slow_percent/100) * self.tau
        # self.y += (1/self.tau * self.dy) * self.demo_dt
        # self.dy += (1/(self.tau)*self.ddy) * self.demo_dt
        # self.ddy += (1/self.tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - self.ddy) + f)) * self.demo_dt

        ## The reality is that RK4 is bad than vanilla euler integration
        def dddy(ddy):
            return (1/tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - ddy) + f))
        self.ddy = integrate_RK4(self.ddy, dddy, self.demo_dt)

        def dy(y):
            return (1/tau * self.dy)
        self.y = integrate_RK4(self.y, dy, self.demo_dt)

        def ddy(dy):
            return (1/tau * self.ddy)
        self.dy = integrate_RK4(self.dy, ddy, self.demo_dt)
        
        return self.y, 1/tau*self.dy, 1/tau**2*self.ddy # all (dof,)

    def trajectory(self, slow_percent=0):
        ''' reconstruct the whole trajectory
        '''
        self.reset_state()
        timesteps = int( np.ceil( self.timesteps * (slow_percent/100 + 1) ) )

        # set up tracking vectors
        y_track = np.zeros([timesteps, self.dof])
        dy_track = np.zeros([timesteps, self.dof])
        ddy_track = np.zeros([timesteps, self.dof])

        for t in range(timesteps):
            # run and record timestep
            if t == 0:
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.y10, self.dy10, self.ddy10
            else:
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.step(t, timesteps, slow_percent)

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
                                          tool_mass=4.,
                                          tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                          tool_cog=[0.0, 0., 0.00],
                                          tool_inertia=[
                                              0., 0., 0., 0., 0., 0.],
                                          load_casadi_fnc=True)
    tau_func = sym_robot.inv_dyn
    tau_max = sym_robot.tau_max
                
    FILE = h5py.File('/home/jiayun/MotionLearning/suboptimal_planner/data/6dof/new_dynamics/box2box_fixedOri/Vali_1e-07_jerk_2.5_mass.hdf5')
    random_index = list(FILE.keys())[2] # 129
    y_demo, dy_demo, ddy_demo = np.array(FILE[random_index]['q']), np.array(FILE[random_index]['qd']), np.array(FILE[random_index]['qdd'])
    length = len(y_demo)
    dt = 0.004
    # DMP
    dmp_demo = FlexDMP(40, demo_len=length, demo_dt=dt, dof=6)  # order=30 is the elbow of Loss curve
    dmp_demo.get_weights(y_demo, dy_demo=dy_demo, ddy_demo=ddy_demo)
    y, dy, ddy = dmp_demo.trajectory(slow_percent=100) # 0.01 * 200 ->  2s
    
    tau = []
    for j in range(len(y)):
        tau.append(sym_robot.inv_dyn(y[j], dy[j], ddy[j]).full().reshape(1,-1))
    tau = np.concatenate(tau)
    violation = np.abs(tau) - np.array(sym_robot.tau_max)
    violation[violation<0.] = 0.
    print(100 * violation.max(axis=0) / np.array(sym_robot.tau_max))

    from scipy.interpolate import CubicSpline
    slow_factor = 0 # in percent
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
        # plt.plot(t_normal, y_demo, '-.r', label="demo q")
        plt.plot(t_normal, y, '-b', label="dmp q")
        plt.plot(t, y_cs, '-c', alpha=0.5, label="post process q")
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc='best')
        plt.grid()

        plt.subplot(3,1,2)
        # plt.plot(t_normal, dy_demo, '-.r', label="demo dq")
        plt.plot(t_normal, dy, '-b', label="dmp dq")
        plt.plot(t, cs(t), '-c', alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc='best')
        plt.grid()
        # plt.savefig("../plots/traj_encoding/DMP_BSpline.jpg", dpi=200)

        plt.subplot(3,1,3)
        # plt.plot(t_normal, ddy_demo, '-.r', label="demo ddq")
        plt.plot(t_normal, ddy, '-b', label="dmp ddq")
        plt.plot(t, cs(t, 1), '-c', alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        labels, ids = np.unique(labels, return_index=True)
        handles = [handles[i] for i in ids]
        plt.legend(handles, labels, loc='best')
        plt.grid()
        plt.savefig("/home/jiayun/Desktop/dmp.jpg", dpi=200)
        plt.show()