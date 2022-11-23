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
            rand_number_2 = int(np.random.uniform(0,datasize))
        A = np.dot(robot_tfs[rand_number_1],np.linalg.inv(robot_tfs[rand_number_2]))
        B = np.dot(pattern_tfs[rand_number_1],np.linalg.inv(pattern_tfs[rand_number_2]))
        alpha.append(SE3.RotToVec(A[:3,:3]))
        beta.append(SE3.RotToVec(B[:3,:3]))
        ta.append(A[:3,3])
        tb.append(B[:3,3])
    Rxinit,txinit = axxb.FCParkSolution(alpha,beta,ta,tb)
    
    rot_res = axxb.IterativeSolutionRot(beta,alpha,sigmaRa,sigmaRb,Rxinit)
    Rxhat, sigmaRx, rot_converged, betahat, alphahat, sigmaRbeta, sigmabeta, sigmaRahat, sigmaRRa = rot_res
    txhat, sigmatx, trans_converged = axxb.IterativeSolutionTrans(betahat, alphahat, ta, tb, Rxhat, sigmaRahat, sigmaRb, sigmata, sigmatb, sigmaRx,sigmaRbeta, txinit.reshape((3,1)), 10)
    if rot_converged and trans_converged:
        Rxlist.append(Rxhat)
        sigmaRx_list.append(sigmaRx)
        txlist.append(txhat.reshape(3))
        sigmatx_list.append(sigmatx)
    else:
        print "Not converged!"," rot_converged ",rot_converged ,"trans_converged ",trans_converged

        
logRx_list = [SE3.RotToVec(Rx) for Rx in Rxlist]
avg_log = np.average(logRx_list,axis=0)
avg_Rx = SE3.VecToRot(avg_log)
inv_avg_Rx = np.linalg.inv(avg_Rx)
avg_tx = np.average(txlist,axis=0)
        
        
xiRx_list =