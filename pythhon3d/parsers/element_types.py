"""
====================================================================================================================
Script :
====================================================================================================================
Defines the types of geometries defining an element, and the order in which vertices are stored in an element
"""

import numpy as np

C_c1d2 = np.array([[0], [1]])
C_c2d3 = np.array([[0, 1], [1, 2], [2, 0]])
C_c2d4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
C_c2d6 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
C_c3d4 = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3]])

C_cf_ref = {
    "c1d2": C_c1d2,
    "c2d3": C_c2d3,
    "c2d4": C_c2d4,
    "c2d6": C_c2d6,
    "c3d4": C_c3d4,
}
