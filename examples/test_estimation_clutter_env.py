import trimesh
import numpy as np
import cope.SE3lib as SE3
import cope.particlelib as ptcl
import cope.transformation as tr
import copy
import time
import pickle


extents    = [0.13,0.1,0.3]
mesh       = trimesh.creation.box(extents)
pkl_file   = open('data/woodstick_w_dict.p', 'rb')
angle_dict = pickle.load(pkl_file)
pkl_file.close()

# ext = [0.05,0.05,0.34]
# other_box = trimesh.creation.box(ext)
# other_box.apply_translation([0.1,0.01,.02])

# rack = trimesh.load_mesh('data/Rack1.ply')
# rack.apply_translation([-0.163,-0.01,-0.17])
# rack.apply_transform(tr.euler_matrix(-3.14/5.,0,3.14/6.))
# rack.apply_translation([0.075,-0.075,0])

# clutter = copy.deepcopy(other_box + rack)


pos_err = 2e-3
nor_err = 5./180.0*np.pi

# Uncertainty & params
sigma0 = np.diag([0.0001,0.0001,0.0001,0.09,0.09,0.09],0)
sigma_desired = 0.09*np.diag([1e-6,1e-6,1e-6,1e-6,1e-6,1e-6],0)
dim = 6 # 6 DOFs
prune_percentage = 0.8
ptcls0 = [np.eye(4)]
M = 6

measurements = [[np.array([-0.06538186,  0.00749609, -0.08090193]),
                 np.array([-0.96346864,  0.26295561, -0.05081864])],
                [np.array([ 0.04767954,  0.06771935, -0.09514227]),
                 np.array([ 0.13469883,  0.97865803,  0.15519239])],
                [np.ar