
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Huy Nguyen <huy.nguyendinh09@gmail.com>
#
# This file is part of python-cope.
#
# python-cope is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# python-cope is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# python-cope. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import scipy.linalg
# Plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def TransformInv(T):
  """
  Calculates the inverse of the input homogeneous transformation.
  
  This method is more efficient than using C{numpy.linalg.inv}, given 
  the special properties of the homogeneous transformations.
  
  @type T: array, shape (4,4)
  @param T: The input homogeneous transformation
  @rtype: array, shape (4,4)
  @return: The inverse of the input homogeneous transformation
  """
  R = T[:3,:3].T
  p = T[:3,3]
  T_inv = np.identity(4)
  T_inv[:3,:3] = R
  T_inv[:3,3] = np.dot(-R, p)
  return T_inv

def TranValidate(T):
  """
  Validate T
  @type T:    array 4x4 
  @param T:   transformation matrix
  """
  raise NotImplementedError


def RotValidate(C):
  raise NotImplementedError


def TranAd(T):
  """
  Compute Adjoint of 4x4 transformation matrix, return a 6x6 matrix
  @type T:    array 4x4 
  @param T:   transformation matrix
  """
  C = T[:3,:3]
  r = T[:3,3]
  AdT = np.zeros([6,6])
  AdT[:3,:3] = C
  AdT[:3,3:] = np.dot(Hat(r),C)
  AdT[3:,3:] = C
  return AdT


def Hat(vec):
  """
  hat operator - return skew matrix (return 3x3 or 4x4 matrix) from vec
  @param vec:   vector of 3 (rotation) or 6 (transformation)
  """
  if vec.shape[0] == 3: # skew from vec
    return np.array([[0,-vec[2],vec[1]],[vec[2],0,-vec[0]],[-vec[1],vec[0],0]])
  elif vec.shape[0] == 6:
    vechat = np.zeros((4,4))
    vechat[:3,:3] = Hat(vec[3:])
    vechat[:3,3] = vec[:3]
    return vechat
  else:
    raise ValueError("Invalid vector length for hat operator\n")


def VecFromSkew(r):
  return np.array([r[2,1],r[0,2],r[1,0]])


def CurlyHat(vec):
  """
  Builds the 6x6 curly hat matrix from the 6x1 input
  @param vec:          a 6x1 vector xi
  @param veccurlyhat:  a 6x6 matrix 
  """
  veccurlyhat = np.zeros((6,6))
  veccurlyhat[:3,:3] = Hat(vec[3:])
  veccurlyhat[:3,3:] = Hat(vec[:3])
  veccurlyhat[3:,3:] = Hat(vec[3:])
  return veccurlyhat


def CovOp1(A):
  """ 
  Covariance operator 1 - eq. 44
  """
  return -np.trace(A)*np.eye(3) + A

 
def CovOp2(A,B):
  """ 
  Covariance operator 2 - eq. 45
  """
  return np.dot(CovOp1(A),CovOp1(B)) + CovOp1(np.dot(B,A))


def TranToVec(T):
  """
  Compute the matrix log of the transformation matrix T
  Convert from T to xi
  @param T:       4x4
  @param return:  return a 6x1 vector in tangent coordinates computed from T.
  """
  C = T[:3,:3]
  r = T[:3,3]
  
  phi = RotToVec(C)
  invJ = VecToJacInv(phi)
  
  rho = np.dot(invJ,r)
  return np.hstack([rho,phi])


# def RotToVec(C):
#   """
#   Compute the matrix log of the rotation matrix C
#   @param C:      3x3
#   @param return: Return a 3x1 vector (axis*angle) computed from C
#   """
#   #rotValidate(C)
#   if(abs(np.trace(C)+1)>1e-10):
#     if(np.linalg.norm(C-np.eye(3))<=1e-10):
#       return np.zeros(3)
#     else:
#       phi = np.arccos((np.trace(C)-1)/2)
#       return VecFromSkew(phi/(2*np.sin(phi))*(C-C.T))
#   else:
#     eigval, eigvect = np.linalg.eig(C)
#     for (i,val) in enumerate(eigval):
#       if abs((val-1)) <= 1e-10:
#         return np.pi*np.real(eigvect[:,i])

def RotToVec(C):
  """
  Compute the matrix log of the rotation matrix C
  @param C:      3x3
  @param return: Return a 3x1 vector (axis*angle) computed from C
  """
  # RotValidate(C)
  epsilon = 0.0001
  epsilon2 = 0.001
  if ((abs(C[0,1]-C[1,0])<epsilon) and (abs(C[0,2]-C[2,0])<epsilon) and (abs(C[1,2]-C[2,1])<epsilon)):
    # singularity found
    # first check for identity matrix which must have +1 for all terms
		# in leading diagonaland zero in other terms
    if ((abs(C[0,1]+C[1,0]) < epsilon2) and (abs(C[0,2]+C[2,0]) < epsilon2) and (abs(C[1,2]+C[2,1]) < epsilon2) and (abs(C[0,0]+C[1,1]+C[2,2]-3) < epsilon2)): # this singularity is identity matrix so angle = 0
      return np.zeros(3) #zero angle, arbitrary axis 
    # otherwise this singularity is angle = 180
    angle = np.pi
    xx = (C[0,0]+1)/2.
    yy = (C[1,1]+1)/2.
    zz = (C[2,2]+1)/2.
    xy = (C[0,1]+C[1,0])/4.
    xz = (C[0,2]+C[2,0])/4.
    yz = (C[1,2]+C[2,1])/4.
    if ((xx > yy) and (xx > zz)): # C[0][0] is the largest diagonal term
      if (xx< epsilon):
        x = 0
        y = np.sqrt(2)/2.
        z = np.sqrt(2)/2.
      else:
        x = np.sqrt(xx)
        y = xy/x
        z = xz/x
    elif (yy > zz): # C[1][1] is the largest diagonal term
      if (yy< epsilon):
        x = np.sqrt(2)/2.
        y = 0
        z = np.sqrt(2)/2.
      else:
        y = np.sqrt(yy)
        x = xy/y
        z = yz/y
    else: # C[2][2] is the largest diagonal term so base result on this
      if (zz< epsilon):
        x = np.sqrt(2)/2.
        y = np.sqrt(2)/2.
        z = 0
      else:
        z = np.sqrt(zz)
        x = xz/z
        y = yz/z
    return angle*np.array((x,y,z))
  s = np.sqrt((C[2,1] - C[1,2])*(C[2,1] - C[1,2])+(C[0,2] - C[2,0])*(C[0,2] - C[2,0])+(C[1,0] - C[0,1])*(C[1,0] - C[0,1])) # used to normalise
  if (abs(s) < 0.001):
    # prevent divide by zero, should not happen if matrix is orthogonal and should be
    # caught by singularity test above, but I've left it in just in case
    s=1 
        
  angle = np.arccos(( C[0,0] + C[1,1] + C[2,2] - 1)/2.)
  x = (C[2,1] - C[1,2])/s
  y = (C[0,2] - C[2,0])/s
  z = (C[1,0] - C[0,1])/s
  return angle*np.array((x,y,z))

def VecToRot(phi):
  """
  Return a rotation matrix computed from the input vec (phi 3x1)
  @param phi: 3x1 vector (input)
  @param C:   3x3 rotation matrix (output)
  """
  tiny = 1e-12
  #check for small angle
  nr = np.linalg.norm(phi)
  if nr < tiny:
    #~ # If the angle (nr) is small, fall back on the series representation.
    # C = VecToRotSeries(phi,10)
    C = np.eye(3)
  else:
    R = Hat(phi)
    C = np.eye(3) + np.sin(nr)/nr*R + (1-np.cos(nr))/(nr*nr)*np.dot(R,R)
  return C


def VecToRotSeries(phi, N):
  """"
  Build a rotation matrix using the exponential map series with N elements in the series 
  @param phi: 3x1 vector
  @param N:   number of terms to include in the series
  @param C:   3x3 rotation matrix (output)
  """
  C = np.eye(3)
  xM = np.eye(3)
  cmPhi = Hat(phi)
  for n in range(N):
    xM = np.dot(xM, cmPhi)/(n+1)
    C = C + xM
  # Project the resulting rotation matrix back onto SO(3)
  C = np.dot(C,np.linalg.inv(scipy.linalg.sqrtm(np.dot(C.T,C))))
  return C


def cot(x):
  return 1./np.tan(x)

  
def VecToJacInv(vec):
  """
  Construction of the 3x3 J^-1 matrix or 6x6 J^-1 matrix.
  @param vec:  3x1 vector or 6x1 vector (input)
  @param invJ: 3x3 inv(J) matrix or 6x6 inv(J) matrix (output)
  """
  tiny = 1e-12
  if vec.shape[0] == 3: # invJacobian of SO3
    phi = vec
    nr = np.linalg.norm(phi)
    if nr < tiny:
      # If the angle is small, fall back on the series representation
      invJSO3 = VecToJacInvSeries(phi,10)
    else:
      axis = phi/nr
      invJSO3 = 0.5*nr*cot(nr*0.5)*np.eye(3) + (1- 0.5*nr*cot(0.5*nr))*axis[np.newaxis]*axis[np.newaxis].T- 0.5*nr*Hat(axis)
    return invJSO3
  elif vec.shape[0] == 6: # invJacobian of SE3
    rho = vec[:3]
    phi = vec[3:]
    
    nr = np.linalg.norm(phi)
    if nr < tiny:
      # If the angle is small, fall back on the series representation
      invJSO3 = VecToJacInvSeries(phi,10)
    else:
      invJSO3 = VecToJacInv(phi)
    Q = VecToQ(vec)
    invJSE3 = np.zeros((6,6))
    invJSE3[:3,:3] = invJSO3
    invJSE3[:3,3:] = -np.dot(np.dot(invJSO3,Q), invJSO3)
    invJSE3[3:,3:] = invJSO3
    return invJSE3
  else:
    raise ValueError("Invalid input vector length\n")


def VecToJacInvSeries(vec,N):
  """
  Construction of the 3x3 J^-1 matrix or 6x6 J^-1 matrix. Series representation.
  @param vec:  3x1 vector or 6x1 vector
  @param N:    number of terms to include in the series
  @param invJ: 3x3 inv(J) matrix or 6x6 inv(J) matrix (output)
  """
  if vec.shape[0] == 3: # invJacobian of SO3
    invJSO3 = np.eye(3)
    pxn = np.eye(3)
    px = Hat(vec)
    for n in range(N):
      pxn = np.dot(pxn,px)/(n+1)
      invJSO3 = invJSO3 + BernoulliNumber(n+1)*pxn
    return invJSO3
  elif vec.shape[0] == 6: # invJacobian of SE3
    invJSE3 =np.eye(6)
    pxn = np.eye(6)
    px = CurlyHat(vec)
    for n in range(N):
      pxn = np.dot(pxn,px)/(n+1)
      invJSE3 = invJSE3 + BernoulliNumber(n+1)*pxn
    return invJSE3
  else:
    raise ValueError("Invalid input vector length\n")


def BernoulliNumber(n):
  """
  Generate Bernoulli number
  @param n:  interger (0,1,2,...)
  """
  from fractions import Fraction as Fr
  if n == 1: return -0.5
  A = [0] * (n+1)
  for m in range(n+1):
    A[m] = Fr(1, m+1)
    for j in range(m, 0, -1):
      A[j-1] = j*(A[j-1] - A[j])
  return A[0].numerator*1./A[0].denominator # (which is Bn)


def VecToJac(vec):
  """ 
  Construction of the J matrix
  @param vec: a 3x1 vector for SO3 or a 6x1 vector for SE3 (input)
  @param J:   a 3x3 J matrix for SO3 or a 6x6 J matrix for SE3 (output)
  """
  tiny = 1e-12
  if vec.shape[0] == 3: # Jacobian of SO3
    phi = vec
    nr = np.linalg.norm(phi)
    if nr < tiny:
      # If the angle is small, fall back on the series representation
      JSO3 = VecToJacSeries(phi,10)
    else:
      axis = phi/nr
      cnr = np.cos(nr)
      snr = np.sin(nr)
      JSO3 = (snr/nr)*np.eye(3) + (1-snr/nr)*axis[np.newaxis]*axis[np.newaxis].T + ((1-cnr)/nr)*Hat(axis)
    return JSO3
  elif vec.shape[0] == 6: # Jacobian of SE3
    rho = vec[:3]
    phi = vec[3:]
    nr = np.linalg.norm(phi)
    if nr < tiny:
      #If the angle is small, fall back on the series representation
      JSE3 = VecToJacSeries(phi,10);
    else:
      JSO3 = VecToJac(phi)
      Q = VecToQ(vec)
      JSE3 = np.zeros((6,6))
      JSE3[:3,:3] = JSO3
      JSE3[:3,3:] = Q
      JSE3[3:,3:] = JSO3
    return JSE3
  else:
    raise ValueError("Invalid input vector length\n")


def VecToJacSeries(vec,N):
  """ 
  Construction of the J matrix from Taylor series
  @param vec: a 3x1 vector for SO3 or a 6x1 vector for SE3 (input)
  @param N:   number of terms to include in the series (input)
  @param J:   a 3x3 J matrix for SO3 or a 6x6 J matrix for SE3 (output)
  """
  if vec.shape[0] == 3: # Jacobian of SO3
    JSO3 = np.eye(3)
    pxn = np.eye(3)
    px = Hat(vec)
    for n in range(N):
      pxn = np.dot(pxn,px)/(n+2)
      JSO3 = JSO3 + pxn
    return JSO3
  elif vec.shape[0] == 6: # Jacobian of SE3
    JSE3 = np.eye(6)
    pxn = np.eye(6)
    px = CurlyHat(vec)
    for n in range(N):
      pxn = np.dot(pxn,px)/(n+2)
      JSE3 = JSE3 + pxn
    return JSE3
  else:
    raise ValueError("Invalid input vector length\n")
  return


def VecToQ(vec):
  """
  @param vec: a 6x1 vector (input)
  @param Q:   the 3x3 Q matrix (output)
  """
  rho = vec[:3]
  phi = vec[3:]
  
  nr = np.linalg.norm(phi)
  if nr == 0:
    nr = 1e-12
  nr2 = nr*nr
  nr3 = nr2*nr
  nr4 = nr3*nr
  nr5 = nr4*nr
  
  cnr = np.cos(nr)
  snr = np.sin(nr)
  
  rx = Hat(rho)