
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

import copy
import math
import numpy as np
import random

import cope.SE3lib as SE3

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.ion()

def Eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def VisualizeCovariances(cov_rot, cov_trans, minx,maxx,miny,maxy):
  plt.subplot(231) # x and y axis
  nstd=1
  alpha = 0.5
  mean = (0,0)
  cov0 = cov_rot

  cov = [[cov0 [0][0],cov0 [0][1]],[cov0 [1][0],cov0 [1][1]]]
  pos = mean

  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip1 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)