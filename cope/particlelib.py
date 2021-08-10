
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