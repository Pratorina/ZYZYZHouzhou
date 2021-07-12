
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