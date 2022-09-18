from trajencoder.bspline.bspline import Bspline
import numpy as np

class AccQuintic(Bspline):
    ''' Over-determined quintic spline
        Fusion the information from trajectory and acceleration via points!
    '''
    def __init__(self, knots, init=0., end=0.):
        ''' given init and end joint position value
            given knots number
            example: ----|----|----|----, knots=3 and there are 4 group of parameters
        '''
        super().__init__(knots, init=init, end=end)
        self.order = 5 # the order of quintic poly

    def get_weights_through_traj(self, T, Y, DDY):
        '''
            Input:
                    T: the time vector
                    Y: joint traj value, has shape (len(T), n), n for number of channel
                    DDY: acceleration traj value, has shape (len(T), n), n for number of channel
        '''
        traj_length, channel = Y.shape
        assert len(T) == traj_length, "The time length dosen't match the trajectory."
        # subsample, including two boundary points
        dt = T[1] - T[0]
        indx = [0] + [i*(traj_length-traj_length%(self.knots+1))//(self.knots+1) for i in range(1, self.knots+1)] + [traj_length-1]
        t = [T[i] for i in indx]
        y = [Y[i].reshape(1,-1) for i in indx]
        y = np.concatenate(y) # y has shape (knots+2, n)
        ddy = [DDY[i].reshape(1,-1) for i in indx]
        ddy = np.concatenate(ddy) # y has shape (knots+2, n)
        self.weights = np.zeros([channel, self.knots+1, self.order])
        for c in range(channel):
            # calc weight for each channel/joints
            weight = self._calc_weight(t, y[:, c], ddy[:, c]).reshape(self.knots+1, self.order)
            self.weights[c, :, :] = weight
        return self.weights

    def get_weights_through_via(self, T, Y, DDY):
        ''' via point interpolating
            Input:
                    T: Time vector has the same dim as Y
                    Y: Via point value
                    DDY: Acc via points
            Output:
                    weights
        '''
        point_num, channel = Y.shape
        assert len(T) == point_num, "The time length dosen't match the trajectory."
        assert point_num == self.knots + 2, "The knots number + 2 should be equal to the point number"
        # subsample, including two boundary points
        self.weights = np.zeros([channel, self.knots+1, self.order]) # e.g 4 for cubic spline
        for c in range(channel):
            # calc weight for each channel/joints
            weight = self._calc_weight(T, Y[:, c], DDY[:, c]).reshape(self.knots+1, self.order)
            self.weights[c, :, :] = weight
        return self.weights

    def _calc_weight(self, t, y, ddy):
        ''' solve the linear system and return the weight for one channel
            here we define an extra init condition qdd = 0, in order to make 5n equations
            input:
                t:  1-d numpy array, (knots+2, dof)
                y:  1-d numpy array, (knots+2, dof), now it's acceleration
                init: float
                end: float
            output:
                weight: 2-d numpy array, shape=[knots+1, 5]
        '''
        width = self.order * (self.knots + 1) # width of A matrix
        A = []
        for i in range(self.knots):
            temp = np.zeros([self.order+2, width]) # start building matrix
            temp[:, i*self.order:i*self.order+2*self.order] = self._build_A_block(t[i], t[i+1])
            A.append(temp)
        A = np.concatenate(A, axis=0)
        # add boundary condition
        A_boundary = np.zeros([self.order-1, width])
        A_boundary[0, -self.order:] = np.array([0., 0., 2, 6*t[-2], 12*t[-2]**2], dtype=float)
        A_boundary[1, -self.order:] = np.array([0., 0., 2, 6*t[-1], 12*t[-1]**2], dtype=float)
        A_boundary[2, -self.order:] = np.array([1., t[-2], t[-2]**2, t[-2]**3, t[-2]**4], dtype=float)
        A_boundary[3, -self.order:] = np.array([1., t[-1], t[-1]**2, t[-1]**3, t[-1]**4], dtype=float)
        # A_boundary[4, :self.order] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # init vel is zero
        # A_boundary[5, -self.order:] = np.array([0., 1., 2*t[0], 3*t[0]**2, 4*t[0]**3], dtype=float)                         # final vel is zero
        A = np.append(A, A_boundary, axis=0)
        # assert A.shape == (5*(self.knots+1), 5*(self.knots+1)), "A might be mistakenly initialized.."
        b = [np.array([y[i], y[i+1], ddy[i], ddy[i+1], 0 ,0, 0], dtype=float) for i in range(0, self.knots)]
        b = np.concatenate(b)
        # add boundary condition
        b_boundary = np.array([ddy[-2], ddy[-1], y[-2], y[-1]], dtype=float)
        b = np.concatenate([b, b_boundary]).reshape(-1, 1)
        #return weight
        weight = np.linalg.pinv(A) @ b
        return weight

    def _build_A_block(self, x, x_next):
        ''' helper function to build the dignonal block of A matrix in _calc_weight
        '''
        A_i =   np.array(
                    [[1.,   x,      x**2,       x**3,       x**4,       0.,         0.,     0.,            0.,                 0.],
                     [1.,   x_next, x_next**2,   x_next**3,   x_next**4,   0.,     0.,    0.,   0.,   0.],
                     [0.,   0.,      2.,       6*x,       12*x**2,           0.,     0.,     0.,         0.,             0.],
                     [0.,   0.,     2.,  6*x_next,  12*x_next**2,      0.,     0.,     0.,         0.,             0.],
                     [0.,   1.,     2*x_next,   3*x_next**2,4*x_next**3,    0.,    -1.,    -2*x_next,  -3*x_next**2,    -4*x_next**3],
                     [1.,   x_next, x_next**2,   x_next**3,   x_next**4,   -1.,     -x_next,    -x_next**2,   -x_next**3,   -x_next**4],
                     [0.,   0.,     0.,         6.,         24*x_next,      0.,     0.,     0.,        -6.,             -24*x_next]], 
                    dtype=float)
        return A_i

    def _polyfunc(self, w_i, t, return_dyddy=False):
        '''
        Helper function to reconstruct traj
        input:
                w_i: the weight of section between two knots, shape (c, 4)
                t:   time scalar
        output:
                y:   poly value in t, shape (c,)
        '''
        t = np.repeat(t.reshape(1,-1), len(self.weights), axis=0)
        y = w_i[:,0:1] + w_i[:,1:2] * t + w_i[:,2:3] * t**2 + w_i[:,3:4] * t**3 + w_i[:,4:5] * t**4
        if return_dyddy:
            dy = w_i[:,1:2] + 2 * w_i[:,2:3] * t + 3*w_i[:,3:4] * t**2 + 4*w_i[:,4:5] * t**3
            ddy = 2 * w_i[:,2:3] + 6*w_i[:,3:4] * t + 12*w_i[:,4:5] * t**2
            return y, dy, ddy
        else:
            return y


if __name__ == "__main__":
    # test run on real data

    import h5py
    from bspline import QuinticSpline
    bs = AccQuintic(18)
    bs_ = QuinticSpline(18)

    File = h5py.File('/home/jiayun/MotionLearning/suboptimal_planner/data/training/2-dof-no-velocity/training_SPL40_D4.hdf5', 'r')
    for label in File.keys():
        t, y, dy, ddy = File[label]["t"], File[label]["q"], File[label]["qd"], File[label]["qdd"]

        bs.get_weights_through_traj(t, y, ddy)
        bs_.get_weights_through_traj(t, y)
        traj, traj_d, traj_dd = bs.trajectory(t[-1]+(t[1]-t[0]), t[1]-t[0], return_dyddy=True)
        traj_, traj_d_, traj_dd_ = bs_.trajectory(t[-1]+(t[1]-t[0]), t[1]-t[0], return_dyddy=True)
        print("There are %i parameters." % bs.weights.size)
        import matplotlib.pyplot as plt
        plt.figure(figsize=[12,8])
        # plt.plot(y[:, 0], '-.r', label="true_q")
        # plt.plot(dy[:, 0], '-.c', label="true_dq")
        plt.plot(ddy[:, 0], '-.c', label="true_ddq")
        # plt.plot(traj[:, 0], 'r', label="q")
        # plt.plot(traj_d[:, 0], 'c', label="dq")
        plt.plot(traj_dd[:, 0], 'b', label="ddq_AccQuintic")
        plt.plot(traj_dd_[:,0], 'r', label='ddq_Quintic')
        plt.legend()
        plt.grid()
        plt.savefig("/home/jiayun/Desktop/spline.jpg", dpi=200)
        plt.show()