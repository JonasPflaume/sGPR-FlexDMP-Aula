import numpy as np
import sys, os, h5py, re
from trajencoder.dmp.dmp import DMP
from trajencoder.flexdmp.flexdmp import FlexDMP
from trajencoder.bspline.bspline import CubicSpline, QuinticSpline
import matplotlib.pyplot as plt
import scipy.interpolate
from casadi_kinodynamics.utils import symbolic_robot

def split(string):
    ''' split the string seperated by |
    '''
    return list(map(float, re.split(r'\|', string)))

def prepare_data(file_addr):
    ''' generate training data X, Y, q (list)
    '''
    f = h5py.File(file_addr, 'r')
    X, Y, q = [], [], []
    for label in f.keys():
        X.append(split(label))
        yi = np.append(np.array(f[label]["weight"]).flatten(), len(f[label]["t"]))
        Y.append(yi)
        q.append(np.array(f[label]["q"]))

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y, q

def prepare_data_via(file_addr, spline, knots=9, return_dyddy=False):
    ''' generate training data for via point predicting
        spline: 
                The class of bspline
    '''
    f = h5py.File(file_addr, 'r')
    X, Y, q = [], [], []
    if return_dyddy:
        dq, ddq = [], []
    bs = spline(knots=knots)
    for label in f.keys():
        X.append(split(label))
        t, y = bs.subsample(f[label]["t"], f[label]["q"])
        yi = np.append(y.flatten(), len(f[label]["t"]))
        Y.append(yi)
        q.append(np.array(f[label]["q"]))
        if return_dyddy:
            dq.append(np.array(f[label]["qd"]))
            ddq.append(np.array(f[label]["qdd"]))

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    if return_dyddy:
        return X, Y, q, dq, ddq
    else:
        return X, Y, q

def prepare_data_flexdmp(file_addr, dmp3, DMP_ORDER=20, return_dyddy=False, return_tau=False, end_mass=False, dt=0.004, dof=2):
    ''' generate training data dmp3
        dmp3:
                the dmp3 class
        end_mass:
                if true the X will add the end effector mass
    '''
    f = h5py.File(file_addr, 'r')
    X, Y, q = [], [], []
    if end_mass:
        M = []
    if return_dyddy:
        dq, ddq = [], []
    if return_tau:
        tau = []
    for label in f.keys():
        if dof == 6:
            X.append(f[label]["x"])
        elif dof == 2:
            X.append(split(label))
        t, y, dy, ddy = f[label]["t"], f[label]["q"], f[label]["qd"], f[label]["qdd"]
        dmp = dmp3(DMP_ORDER, len(t), dt, dof=dof)
        yi = dmp.get_weights(y, dy_demo=dy, ddy_demo=ddy)
        yi = np.append(yi.reshape(-1,), float(len(t)))
        Y.append(yi)
        q.append(np.array(f[label]["q"]))
        if return_dyddy:
            dq.append(np.array(f[label]["qd"]))
            ddq.append(np.array(f[label]["qdd"]))
        if end_mass:
            M.append(np.array(f[label]["mass"]))
        if return_tau:
            tau.append(np.array(f[label]["tau"]))

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    f.close()
    if end_mass:
        M = np.array(M).reshape(-1,1)
        X = np.concatenate([X, M], axis=1)

    if return_dyddy and return_tau:
        return X, Y, q, dq, ddq, tau
    elif return_dyddy and not return_tau:
        return X, Y, q, dq, ddq
    elif not return_dyddy and return_tau:
        return X, Y, q, tau
    else:
        return X, Y, q

def prepare_acc_data(file_addr):
    ''' generate training data X, Y, q (list)
    '''
    f = h5py.File(file_addr, 'r')
    X, Y = [], []
    for label in f.keys():
        start_end = np.array(split(label)).reshape(1,-1)
        q = f[label]["q"]
        qd = f[label]["qd"]
        qdd = f[label]["qdd"]
        tile = np.ones(len(q)).reshape(-1,1)
        repeat_start_end = tile*start_end
        x = np.concatenate([repeat_start_end, q, qd], axis=1)
        X.append(x)
        Y.append(qdd)

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y

#####################################################################################
def DMP_reconstruct(y, DMP_ORDER, start, goal, length, dof, dt=0.004, DMP_dt=0.01, return_dyddy=False):
    ''' length: length of original traj with dt=0.004
        reconstruct DMP trajectory from prameters: y
        dt is the original traj sampling inerval, use this to get total time
        then use DMP_dt to get the timesteps for reconstructed traj 
    '''
    t_total = length * dt
    t1 = np.linspace(0, t_total, length)
    t2 = np.linspace(0, t_total, int(t_total/DMP_dt))
    temp = [DMP(DMP_ORDER, dt, length) for _ in range(dof)]
    res_q, res_dq, res_ddq = [], [], []
    for i in range(dof):
        temp[i].set_weight(y[:,i], start[i], goal[i])
        q, qd, qdd = temp[i].trajectory()
        res_q.append(q.reshape(-1,1))
        res_dq.append(qd.reshape(-1,1))
        res_ddq.append(qdd.reshape(-1,1))
    res_q = np.concatenate(res_q, axis=1)
    res_dq = np.concatenate(res_dq, axis=1)
    res_ddq = np.concatenate(res_ddq, axis=1)
    if return_dyddy:
        return res_q, res_dq, res_ddq, t1, t2
    else:
        return res_q, t1, t2 # orignal traj time 0.004, reconstruct time DMP 0.01

def calc_loss(pred, Yv, Trueth, DMP_ORDER, dof=2, dt=0.004, return_dyn_loss=False, tau_func=None, tau_max=None):
    ''' get the
            1. predicting time error
            2. predicting shape error compared with the original traj
    '''
    data_num = len(Yv)
    error_time = 0
    error_target_q = 0
    error_dyn_loss = 0
    dyn_violation_count = 0
    for i in range(data_num):
        pred_y = pred[i, :2*DMP_ORDER].reshape(DMP_ORDER, dof)
        target_y = Yv[i, :2*DMP_ORDER].reshape(DMP_ORDER, dof)
        pred_length = int(pred[i, -1])
        target_length = int(Yv[i, -1])
        q = Trueth[i]
        if not return_dyn_loss:
            res_q_, t1_, t2_ = DMP_reconstruct(pred_y, DMP_ORDER, q[0], q[-1], target_length, dof)
            traj_gen_1q = scipy.interpolate.interp1d(t2_, res_q_[:,0])
            traj_gen_2q = scipy.interpolate.interp1d(t2_, res_q_[:,1])
            res_q_inter = np.zeros_like(q)
            for j, t in enumerate(t1_):
                res_q_inter[j, 0] = traj_gen_1q(t)
                res_q_inter[j, 1] = traj_gen_2q(t)
        else:
            assert tau_func != None and tau_max != None, "when check dyna, you need input the casadi function and max tau to calc the violation."
            from check_dyn import get_traj_tau
            res_q_, res_dq, res_ddq, t1_, t2_ = DMP_reconstruct(pred_y, DMP_ORDER, q[0], q[-1], target_length, dof, return_dyddy=return_dyn_loss)
            traj_gen_1q = scipy.interpolate.interp1d(t2_, res_q_[:,0])
            traj_gen_2q = scipy.interpolate.interp1d(t2_, res_q_[:,1])
            traj_gen_1dq = scipy.interpolate.interp1d(t2_, res_dq[:,0])
            traj_gen_2dq = scipy.interpolate.interp1d(t2_, res_dq[:,1])
            traj_gen_1ddq = scipy.interpolate.interp1d(t2_, res_ddq[:,0])
            traj_gen_2ddq = scipy.interpolate.interp1d(t2_, res_ddq[:,1])

            res_q_inter = np.zeros_like(q)
            res_dq_inter = np.zeros_like(q)
            res_ddq_inter = np.zeros_like(q)
            for j, t in enumerate(t1_):
                res_q_inter[j, 0] = traj_gen_1q(t)
                res_q_inter[j, 1] = traj_gen_2q(t)
                res_dq_inter[j, 0] = traj_gen_1dq(t)
                res_dq_inter[j, 1] = traj_gen_2dq(t)
                res_ddq_inter[j, 0] = traj_gen_1ddq(t)
                res_ddq_inter[j, 1] = traj_gen_2ddq(t)

            tau = get_traj_tau(tau_func, res_q_inter, res_dq_inter, res_ddq_inter)
            plt.figure()
            plt.plot(t1_, res_ddq_inter)
            dyna_loss_temp = np.abs(tau) - tau_max
            dyna_violation_i = (dyna_loss_temp[dyna_loss_temp > 0.]).sum() * dt
            if dyna_violation_i > 0:
                dyn_violation_count += 1
                print("The {}the traj violate the dynamic constraints!".format(i))
            error_dyn_loss += dyna_violation_i
        error_time += np.abs(pred_length*dt - target_length*dt)
        error_target_q += np.abs(q - res_q_inter).sum() * dt
        
    if return_dyn_loss:
        error_dyn_loss /= len(pred)
        error_target_q /= len(pred)
        error_time /= len(pred)
        print("The rate of dyna violation: ", (dyn_violation_count/data_num)*100, "%")
        return error_time, error_target_q, error_dyn_loss
    else:
        error_target_q /= len(pred)
        error_time /= len(pred)
        return error_time, error_target_q

#########################################################################
def spline_reconstruct(y, SPL_ORDER, length, dt=0.004):
    ''' Input:
                y:          the weight, shape (dof, knots+1, 4)
                order:      encode order
                length:     integer, time length

    '''
    bs = QuinticSpline(knots=SPL_ORDER)
    bs.set_weight(y)
    res_q = bs.trajectory(length*dt, dt)
    t = np.linspace(0, dt*length, length)
    return res_q, t

def calc_loss_spl(pred, Yv, Trueth, SPL_ORDER, dof=2, dt=0.004):
    ''' get the
            1. predicting time error
            2. predicting shape error compared with the original traj
    '''
    data_num = len(Yv)
    error_time = 0
    error_target_q = 0
    for i in range(data_num):
        pred_y = pred[i, :-1].reshape(dof, SPL_ORDER+1, 4)
        target_y = Yv[i, :-1].reshape(dof, SPL_ORDER+1, 4)
        pred_length = int(pred[i, -1])
        target_length = int(Yv[i, -1])
        q = Trueth[i]
        res_q, t = spline_reconstruct(pred_y, SPL_ORDER, pred_length, dt=0.004)
        error_time += np.abs(pred_length*dt - target_length*dt) / data_num
        if len(q) > len(res_q):
            q = q[:len(res_q)]
        else:
            res_q = res_q[:len(q)]
        error_target_q += np.abs(q - res_q).sum() / len(q)
        
    return error_time, error_target_q

############################################################################
def spline_reconstruct_via(y, spline, SPL_ORDER, length, dt=0.004, return_dyddy=False):
    ''' Input:
                y:          via points, shape (dof, knots+2)
                spline:     spline class
                order:      encode order
                length:     integer, time length

    '''
    bs = spline(knots=SPL_ORDER)
    T = np.linspace(0, dt*length, length)
    indx = [0] + [i*(length-length%(SPL_ORDER+1))//(SPL_ORDER+1) for i in range(1, SPL_ORDER+1)] + [length-1]
    t = np.array([T[i] for i in indx])
    bs.get_weights_through_via(t, y)
    if return_dyddy:
        res_q, res_dq, res_ddq = bs.trajectory(length*dt, dt, return_dyddy=return_dyddy)
        return res_q, res_dq, res_ddq, T
    else:
        res_q = bs.trajectory(length*dt, dt, return_dyddy=return_dyddy)
        return res_q, T

def calc_loss_spl_via(pred, Xv, Yv, Trueth, spline, SPL_ORDER, return_dyn_loss=False, return_dyn_info=False, tau_func=None, tau_max=None, dof=2, dt=0.004):
    ''' get the
            1. predicting time error
            2. predicting shape error compared with the original traj
    '''
    data_num = len(pred)
    error_time = 0
    error_target_q = 0
    error_dyn_loss = 0
    dyn_violation_count = 0
    dyn_violation_peak = np.zeros(dof)
    dyn_violation_peak_index = np.zeros(dof)
    for i in range(data_num):
        pred_y = pred[i, :-1].reshape(SPL_ORDER, dof)
        pred_y = np.concatenate([Xv[i, :dof].reshape(1, -1), pred_y], axis=0)
        pred_y = np.concatenate([pred_y, Xv[i, dof:].reshape(1, -1)], axis=0)

        pred_length = int(pred[i, -1])
        target_length = int(Yv[i, -1])

        q = Trueth[i]
        if not return_dyn_loss:
            res_q, t = spline_reconstruct_via(pred_y, spline, SPL_ORDER, pred_length, dt=0.004)
        else:
            assert tau_func != None and tau_max != None, "when check dyna, you need input the casadi function and max tau to calc the violation."
            res_q, dq, ddq, t = spline_reconstruct_via(pred_y, spline, SPL_ORDER, pred_length, dt=0.004, return_dyddy=True)
            from check_dyn import get_traj_tau
            tau = get_traj_tau(tau_func, res_q, dq, ddq)
            dyna_loss_temp = np.abs(tau) - tau_max
            dyna_violation_i = (dyna_loss_temp[dyna_loss_temp > 0.]).sum() * dt
            dyna_violation_max = np.max(dyna_loss_temp, axis=0)
            # set the peak violation
            for k in range(dof):
                if dyn_violation_peak[k] < dyna_violation_max[k]:
                    dyn_violation_peak[k] = dyna_violation_max[k]
                    dyn_violation_peak_index[k] = i
            if dyna_violation_i > 0:
                dyn_violation_count += 1
            error_dyn_loss += dyna_violation_i
        error_time += np.abs(pred_length*dt - target_length*dt)
        if len(q) > len(res_q):
            q = q[:len(res_q)]
        else:
            res_q = res_q[:len(q)]
        error_target_q += np.abs(q - res_q).sum() * dt
    
    if return_dyn_loss:
        if return_dyn_info:
            dyna_info = {}
            error_dyn_loss /= len(pred)
            error_target_q /= len(pred)
            error_time /= len(pred)
            print("The rate of dyna violation: ", (dyn_violation_count/data_num)*100, "%")
            print("The peak dyna violation: ", dyn_violation_peak)
            print("Peak dyna violation occurs at: ", dyn_violation_peak_index)
            dyna_info["vio_rate"] = (dyn_violation_count/data_num)*100
            dyna_info["vio_peak"] = dyn_violation_peak
            return error_time, error_target_q, error_dyn_loss, dyna_info
        else:
            error_dyn_loss /= len(pred)
            error_target_q /= len(pred)
            error_time /= len(pred)
            print("The rate of dyna violation: ", (dyn_violation_count/data_num)*100, "%")
            print("The peak dyna violation: ", dyn_violation_peak)
            print("Peak dyna violation occurs at: ", dyn_violation_peak_index)
            return error_time, error_target_q, error_dyn_loss
    else:
        error_target_q /= len(pred)
        error_time /= len(pred)
        return error_time, error_target_q

#############################################################
def flexdmp_reconstruct(PARAM, initial, goal, length, DMP_ORDER, dof, dt=0.004, slowpercent=0):
    ''' Input:
                PARAM:          Predicting parameters, shape (dof, DMP_ORDER)
                DMP_ORDER:      Order of FlexDMP
                length:         integer, predicted time length
                dof:            Dof of robot
                initial, goal:  The start and goal of traj
    '''
    assert PARAM.shape == (dof, DMP_ORDER), "Input the right shape of parameters!"
    dmp = FlexDMP(DMP_ORDER, length, dt, dof=dof)

    dmp.set_weight(PARAM, initial, goal)
    res_q, res_dq, res_ddq = dmp.trajectory(slow_percent=slowpercent)
    return res_q, res_dq, res_ddq

from check_dyn import get_traj_tau

def calc_loss_flexdmp(pred, Xv, Yv, DMP_ORDER, dt=0.004, dof=6, prolong=False):
    ''' get the
            1. predicting time error
            2. predicting shape error compared with the original traj
        Input:
                pred:      The predicted parameters of dmp, shape (point_number, dof x DMP_ORDER + 1)
                Xv:         Validating input set, shape (point_number, feature_number)
                Yv:         Validating output set, shape (point_number, output_number)
                Trueth:     Ground trueth trajectories, list with length point_number, elements have shape (traj_length, dof)
                DMP_ORDER:  The order of DMP
    '''
    time_loss = []
    torque_violation = []

    for index in range(len(Xv)):
        sym_robot = symbolic_robot.symbolic_robot(robot_name='IRB1100_4_058',
                                            tool_mass=Xv[index, -1],
                                            tool_frame=[0., 0., 0.0, 0., 0., 0.],
                                            tool_cog=[0.0, 0., 0.0],
                                            tool_inertia=[
                                                0., 0., 0., 0., 0., 0.],
                                            load_casadi_fnc=True)
        tau_func = sym_robot.inv_dyn
        tau_max = sym_robot.tau_max
        
        PARAM = pred[index,:-1].reshape(dof, DMP_ORDER)
        initial = Xv[index,:dof]
        goal = Xv[index,dof:2*dof]
        init_length = int( pred[index,-1] )
        
        q, dq, ddq = flexdmp_reconstruct(PARAM, initial, goal, init_length, DMP_ORDER, dof)
        if prolong:
            tau = get_traj_tau(tau_func, q, dq, ddq)
            violation = np.abs(tau) - np.array(tau_max)
            violation[violation<0.] = 0.0
            violation = np.max(violation, axis=0)
            max_violation_percent = (violation / np.array(tau_max) * 100).max()
            counter = 0
            while max_violation_percent > 5.:
                counter += 1
                print("Prolong percent:", 2**counter)
                q, dq, ddq = flexdmp_reconstruct(PARAM, initial, goal, init_length, DMP_ORDER, dof, slowpercent=2**counter)
                tau = get_traj_tau(tau_func, q, dq, ddq)
                violation = np.abs(tau) - np.array(tau_max)
                violation[violation<0.] = 0.0
                violation = np.max(violation, axis=0)
                max_violation_percent = (violation / np.array(tau_max) * 100).max()
            torque_violation.append( violation.copy().reshape(1,-1) )
            time_loss.append( np.abs(init_length - Yv[index, -1]) * dt )
        else:
            tau = get_traj_tau(tau_func, q, dq, ddq)
            violation = np.abs(tau) - np.array(tau_max)
            violation[violation<0.] = 0.0
            violation = np.max(violation, axis=0)
            torque_violation.append( violation.copy().reshape(1,-1) )
            time_loss.append( np.abs(init_length - Yv[index, -1]) * dt )
        
    time_loss = np.array( time_loss )
    torque_violation = np.concatenate( torque_violation )
    peak_violation = np.max( torque_violation, axis=0 )
    peak_violation_index = []
    for d in range(dof):
        index_d = np.argmax( torque_violation[:,d] )
        peak_violation_index.append( index_d )
    peak_violation_index = np.array( peak_violation_index )
    return time_loss, torque_violation, peak_violation, peak_violation_index
