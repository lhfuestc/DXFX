# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:32:27 2018

@author: 144-Tesla-K20
"""

import numpy as np

a = np.array([[1,2,3],[1,2,6]])
b = np.array([[1,2,3],[1,7,6]])

assert (a == b).all(), "a and b must be equal"
