
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Huy Nguyen <huy.nguyendinh09@gmail.com>
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

import numpy  as np
import trimesh
import scipy as sp
from scipy.stats import norm
import cope.SE3lib as SE3
import bisect
import cope.transformation as tr
import random
import time
import copy
import matplotlib.pyplot as plt


class Region(object):
  def __init__(self, particles, delta_rot,delta_trans):
    self.particles = particles #List of particles (transformations)
    self.delta_rot = delta_rot
    self.delta_trans = delta_trans

def IsInside(point,center,radius):
  if np.linalg.norm(point-center) < radius:
    return True
  return False


def EvenDensityCover(region, M):
  '''Input: Region V_n - sampling region represented as a union of neighborhoods, M - number of particles to sample per neighborhood
  Output: a set of particles that evenly cover the region (the new spheres will have analogous shape to the region sigma)
  '''
  particles = []
  num_spheres = len(region.particles)
  delta_rot = region.delta_rot
  delta_trans = region.delta_trans
  for i  in range(num_spheres):
    center_particle = region.particles[i]
    center_vec_rot =  SE3.RotToVec(center_particle[:3,:3])
    center_vec_trans = center_particle[:3,3]
    num_existing = 0
    for p in particles:
      if IsInside(SE3.RotToVec(p[:3,:3]),center_vec_rot,delta_rot) and IsInside(p[:3,3],center_vec_trans,delta_trans):
        num_existing += 1
    for m in range(M-num_existing):
      count = 0
      accepted = False
      while not accepted and count < 5:
        new_vec_rot = np.random.uniform(-1,1,size = 3)*delta_rot + center_vec_rot
        new_vec_trans = np.random.uniform(-1,1,size = 3)*delta_trans + center_vec_trans
        count += 1
        accepted = True
        for k in range(i-1):
          previous_center = region.particles[k]
          previous_vec_rot = SE3.RotToVec(previous_center[:3,:3])
          previous_vec_trans = previous_center[:3,3]
          if IsInside(SE3.RotToVec(p[:3,:3]),previous_vec_rot,delta_rot) and IsInside(p[:3,3],previous_vec_trans,delta_trans):
            accepted = False
            break
      if accepted:
        new_p = np.eye(4)
        new_p[:3,:3] = SE3.VecToRot(new_vec_rot)
        new_p[:3,3] = new_vec_trans
        particles.append(new_p)
  return particles

def normalize(weights):
  norm_weights = np.zeros(len(weights))
  sum_weights = np.sum(weights)
  if sum_weights == 0:
    return np.ones(len(weights))/len(weights)
  for i in range(len(weights)):
    norm_weights[i] = weights[i]/sum_weights
  return norm_weights

def ComputeNormalizedWeightsB(mesh,sorted_face,particles,measurements,pos_err,nor_err,tau):
  num_particles = len(particles)
  new_weights = np.zeros(num_particles)
  for i in range(len(particles)):
    T = np.linalg.inv(particles[i])
    D = copy.deepcopy(measurements)
    for d in D:
      d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
      d[1] = np.dot(T[:3,:3],d[1])
    total_energy = sum([FindminimumDistanceMeshOriginal(mesh,sorted_face,measurement,pos_err,nor_err)**2 for measurement in D])
    new_weights[i] = (np.exp(-0.5*total_energy/tau))
  return normalize(new_weights)

def ComputeNormalizedWeights(mesh,sorted_face,particles,measurements,pos_err,nor_err,tau):
  num_particles = len(particles)
  new_weights = np.zeros(num_particles)
  for i in range(len(particles)):
    T = np.linalg.inv(particles[i])
    D = copy.deepcopy(measurements)
    for d in D:
      d[0] = np.dot(T[:3,:3],d[0]) + T[:3,3]
      d[1] = np.dot(T[:3,:3],d[1])
    total_energy = sum([FindminimumDistanceMesh(mesh,sorted_face,measurement,pos_err,nor_err)**2 for measurement in D])
    new_weights[i] = (np.exp(-0.5*total_energy/tau))
  
  return normalize(new_weights)


def FindminimumDistanceMesh(mesh,sorted_face,measurement,pos_err,nor_err):
    ref_vec = sorted_face[2]
    sorted_angle = sorted_face[1]
    face_idx = sorted_face[0]
    angle =  np.arccos(np.dot(measurement[1],ref_vec))
    idx = bisect.bisect_right(sorted_angle,angle)
    if idx >= len(sorted_angle):
      up_bound = idx
    else:
      up_bound = idx + bisect.bisect_right(sorted_angle[idx:],sorted_angle[idx]+sorted_angle[idx]-angle+nor_err)
    if idx == 0:
      low_bound = 0
    else:
      low_bound = bisect.bisect_left(sorted_angle[:idx],sorted_angle[idx-1]-(sorted_angle[idx-1]-angle)-nor_err)-1
    dist = []
    for i in range(low_bound,up_bound):
        A,B,C = mesh.faces[face_idx[i]]
        dist.append(CalculateDistanceFace([mesh.vertices[A],mesh.vertices[B],mesh.vertices[C],mesh.face_normals[face_idx[i]]],measurement,pos_err,nor_err))
    return min(dist)

def FindminimumDistanceMeshOriginal(mesh,sorted_face,measurement,pos_err,nor_err):