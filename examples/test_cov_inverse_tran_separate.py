import numpy as np
import cope
import cope.transformation as tr


ksamples = 500
T =  tr.random_rotation_matrix()
T[:3,3] = tr.random_vector(3)
R = T[:3,:3]
Rinv = np.linalg.inv(R)
t = T[:3,3]
tinv = -np.dot(Rinv,t)
scale = 1e-3
sigmat = scale*np.diag(tr.random_vector(3))#(0.1,0.2,0.5))
sigmaR = scale*np.diag(tr.random_vector(3))#(0.5,0.2,0.3))

xi_vec_Rinv = []
xi_tinv = []

for i in range(ksamples):
    xisampleR = np.random.multivariate_normal(np.zeros(3),sigmaR)
