import numpy as np
import h5py, os

# Just to show how to use the collected data

curr_path = os.path.dirname(os.path.abspath(__file__))

f = h5py.File(os.path.join(curr_path, "training.hdf5"), 'r')

print(len(f.keys()))