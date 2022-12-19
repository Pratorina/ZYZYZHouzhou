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
an