import numpy  as np
import cope
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

starttime = time.time()
tiny = 1e-5

T1 = np.eye(4)
sigma1 = np.diag([tiny,tiny,tiny,tiny,tiny,tiny],0)

T2 = np.eye(4)
T2[:3,3] = np.array([0,0.15,0])
sigma2 = np.diag([tiny,tiny,tiny,0.01,tiny,0.01],0)

T3 = np.eye(4)
T3[:3,3] = np.array([0,0,-0.03])
sigma3 = np.diag([tiny,tiny,tiny,tiny,tiny,0.1],0)

T4 = np.eye(4)
T4[:3,3] = np.array([0,0.14,0])
sigma4 = np.diag([tiny,tiny,tiny,tiny,tiny,tiny],0)



T12, sigma12 = cope