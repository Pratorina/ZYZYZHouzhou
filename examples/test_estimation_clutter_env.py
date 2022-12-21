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
# other_box = trimesh.creation.box(