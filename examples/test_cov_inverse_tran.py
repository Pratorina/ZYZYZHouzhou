import numpy as np
import cope
import cope.transformation as tr


ksamples = 500
T =  tr.random_rotation_matrix()
T[:3,3] = tr.random_vector(3)
vec = cope.TranToVec(T)
scale = 1e-3
sigma = scale