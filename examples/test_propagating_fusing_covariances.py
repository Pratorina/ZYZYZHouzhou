import numpy  as np
import cope
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

starttime = time.time()
tiny = 1e-5

T1 = np.eye(4)
sigma1 = np.diag([tiny,tiny,ti