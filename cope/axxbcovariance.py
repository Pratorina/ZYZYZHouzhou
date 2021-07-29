
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
  ax.add_artist(ellip1)
  plt.axis([minx,maxx,miny,maxy])
  plt.xlabel(r'${\bf{\xi}}_{\bf{R} x} (rad)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{R} y} (rad)$',fontsize=20, labelpad=-8)
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  ax.set(aspect='equal')


  plt.subplot(234)
  mean = (0,0)
  cov = cov_trans
  cov = [[cov [0][0],cov [0][1]],[cov [1][0],cov [1][1]]]
  pos=mean
  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip2 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)
  ax.add_artist(ellip2)
  plt.axis([minx,maxx,miny,maxy])
  plt.xlabel(r'${\bf{\xi}}_{\bf{t} x} (mm)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{t} y} (mm)$',fontsize=20, labelpad=-8)
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  ax.set(aspect='equal')
 ############################################################

  plt.subplot(232)
  cov = cov_rot
  cov = [[cov0 [1][1],cov0 [1][2]],[cov0 [2][1],cov0 [2][2]]]
  pos = mean

  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip1 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)
  ax.add_artist(ellip1)
  plt.axis([minx,maxx,miny,maxy])
  plt.xlabel(r'${\bf{\xi}}_{\bf{R} y} (rad)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{R} z} (rad)$',fontsize=20, labelpad=-8)  
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  ax.set(aspect='equal')

  plt.subplot(235)
  mean = (0,0)
  cov = cov_trans
  cov = [[cov [1][1],cov [1][2]],[cov [2][1],cov [2][2]]]
  pos=mean
  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip2 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)
  ax.add_artist(ellip2)
  plt.axis([minx,maxx,miny,maxy])
  plt.xlabel(r'${\bf{\xi}}_{\bf{t} y} (mm)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{t} z} (mm)$',fontsize=20, labelpad=-8)
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  ax.set(aspect='equal')

  ###################################################################

  plt.subplot(233)
  cov = cov_rot
  cov = [[cov0 [0][0],cov0 [0][2]],[cov0 [2][0],cov0 [2][2]]]
  pos = mean

  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip1 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)
  ax.add_artist(ellip1)
  plt.axis([minx,maxx,miny,maxy])
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  plt.xlabel(r'${\bf{\xi}}_{\bf{R} x} (rad)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{R} z} (rad)$',fontsize=20, labelpad=-8)
  ax.set(aspect='equal')
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)

  plt.subplot(236)
  mean = (0,0)
  cov = cov_trans
  cov = [[cov [0][0],cov [0][2]],[cov [2][0],cov [2][2]]]

  pos=mean
  ax = plt.gca()
  vals, vecs = Eigsorted(cov)
  theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
  # Width and height are "full" widths, not radius
  width, height = 2 * nstd * np.sqrt(vals)
  ellip2 = Ellipse(xy=pos, width=width, height=height, angle=theta,alpha=0.5,color='red', linewidth=2, fill=False)
  ax.add_artist(ellip2)
  plt.axis([minx,maxx,miny,maxy])
  plt.xticks(np.arange(minx, maxx+maxx/2, (maxx-minx)/2))
  plt.yticks(np.arange(miny, maxy+maxy/2, (maxy-miny)/2))
  plt.xlabel(r'${\bf{\xi}}_{\bf{t} x}(mm)$',fontsize=20, labelpad=8)
  plt.ylabel(r'${\bf{\xi}}_{\bf{t} z}(mm)$',fontsize=20, labelpad=-8)
  ax.set(aspect='equal')
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
  # plt.legend(handles=[ellip1],loc='upper center',ncol=2, bbox_to_anchor=(0.5,1.5))
  return True

def VisualizeRealEstCov(cov_real, cov_est, minx,maxx,miny,maxy,param):
    if param=='rot':
        subplotnum = 230
    if param=='trans':
        subplotnum = 233
    # Compare y z
    nstd=1
    alpha = 0.5
    mean = (0,0)