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

# clutter = copy.deepcopy