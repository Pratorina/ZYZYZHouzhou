#!/usr/bin/env python
import copy
import math
import numpy as np
import random
import pickle
import cope.SE3lib as SE3
import cope.axxbcovariance as axxb
import matplotlib.pyplot as plt

# Read data files
filename = "data/pattern_tfs"
pattern_tfs =  pickle.load(open( filename, "rb" ) )
filename = "data/robot_tfs"
robot_tfs =  pickle.load(open( filename, "rb" ) )

sigmaA = 1e-10*np.diag((1, 1, 1, 1, 1, 1))
sigmaRa = sigmaA[3:,3:]
sigmata = sigmaA[:3,:3]

sigmaRb = np.array([[  4.15625435e-05,  -2.88693145e-05,  -6.06526440e-06],
                    [ -2.88693145e-05,   3.20952008e-04,  -1.44817304e-06],
                    [ -6.06526440e-06,  -1.44817304e-06,   1.43937081e-05]])
sigmatb = np.array([[  1.95293655e-04,   2.12627214e-05,  -1.06674886e-05],
                    [  2.12627214e-05,   4.44314426e-05,   3.86787591e-06],
                    [ -1.06674886e-05,   3.86787591e-06,   2.13069579e-05]])

datasize = len(pattern_tfs)
ksamples = 30
iters = 500
Rxlist = []
sigmaRx_list = []
txlist = []
sigmatx_list = []
for n in range(iters):
    alpha = []
    beta = []
    ta = []
    tb = []
    # Generate data-A and B matrices
    for i in range(ksamples):
        # note this
        rand_number_1 = int(np.random.uniform(0,datasize))
        rand_number_2 = int(np.random.uniform(0,datasize))
        while rand_number_1==rand_number_2:
    