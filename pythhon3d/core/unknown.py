import numpy as np
from typing import List
from numpy import ndarray as Mat

from core.face import Face
from core.cell import Cell


class Unknown:
    def __init__(
        self,
        problem_dimension: int,
        field_dimension: int,
        cell_polynomial_order: int,
        face_polynomial_order: int,
        symmetric_gradient: bool = False,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Attributes :
        ================================================================================================================
        
        """
        if problem_dimension == field_dimension:
            if not symmetric_gradient:
                if problem_dimension == 1:
                    indices = [(0, 0)]
                elif problem_dimension == 2:
                    indices = [(0, 0), (1, 1), (0, 1), (1, 0)]
                elif problem_dimension == 3:
                    indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1), (2, 1), (2, 0), (1, 0)]
                else:
                    raise ValueError("the field_dimension is not correct")
            else:
                if problem_dimension == 1:
                    indices = [(0, 0)]
                elif problem_dimension == 2:
                    indices = [(0, 0), (1, 1), (0, 1)]
                elif problem_dimension == 3:
                    indices = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
                else:
                    raise ValueError("the field_dimension is not correct")
        elif problem_dimension > field_dimension:
            if problem_dimension == 1:
                indices = [(0, 0)]
            elif problem_dimension == 2:
                indices = [(0, 0), (0, 1)]
            elif problem_dimension == 3:
                indices = [(0, 0), (0, 1), (0, 2)]
            else:
                raise ValueError("the field_dimension is not correct")
        # --------------------------------------------------------------------------------------------------------------
        self.problem_dimension = problem_dimension
        self.field_dimension = field_dimension
        self.indices = indices
        self.symmetric_gradient = symmetric_gradient
        self.cell_polynomial_order = cell_polynomial_order
        self.face_polynomial_order = face_polynomial_order
        # self.integration_order = 2 * max(face_polynomial_order, cell_polynomial_order)
        self.integration_order = max(2 * (face_polynomial_order + 1), 2 * cell_polynomial_order)
        self.integration_order = 2 * (face_polynomial_order + 1)
        # self.symmetric_gradient = symmetric_gradient
        # self.field_dimension = field_dimension
