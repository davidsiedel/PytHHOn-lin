from core.face import Face
from core.cell import Cell
from core.integration import Integration
from core.unknown import Unknown
from bases.basis import Basis
from bases.monomial import ScaledMonomial

import numpy as np
from typing import List
from numpy import ndarray as Mat


class Operator:
    def __init__(
        self, local_gradient_operator: Mat, local_stabilization_operator: Mat, local_mass_operator: Mat,
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
        self.local_gradient_operator = local_gradient_operator
        self.local_stabilization_operator = local_stabilization_operator
        self.local_mass_operator = local_mass_operator

    def get_local_problem_size(self, faces: List[Face], cell_basis: Basis, face_basis: Basis, unknown: Unknown):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        number_of_faces = len(faces)
        local_problem_size = (
            cell_basis.basis_dimension * unknown.field_dimension
            + number_of_faces * face_basis.basis_dimension * unknown.field_dimension
        )
        return local_problem_size

    @staticmethod
    def get_vector_to_face(cell: Cell, face: Face) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        p = face.reference_frame_transformation_matrix
        vector_to_face = (p @ (cell.centroid - face.centroid).T).T
        return vector_to_face

    @staticmethod
    def get_swaped_face_reference_frame_transformation_matrix(cell: Cell, face: Face) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        p = face.reference_frame_transformation_matrix
        problem_dimension = p.shape[1]
        # --------------------------------------------------------------------------------------------------------------
        # 2d faces in 3d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 3:
            swaped_reference_frame_transformation_matrix = np.array([p[0], p[1], -p[2]])
        # --------------------------------------------------------------------------------------------------------------
        # 1d faces in 2d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 2:
            swaped_reference_frame_transformation_matrix = np.array([p[0], -p[1]])
        # --------------------------------------------------------------------------------------------------------------
        # 0d faces in 1d cells
        # --------------------------------------------------------------------------------------------------------------
        if problem_dimension == 1:
            swaped_reference_frame_transformation_matrix = np.array([-p[0]])
        return swaped_reference_frame_transformation_matrix

    @staticmethod
    def get_face_passmat(cell: Cell, face: Face):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        vector_to_face = Operator.get_vector_to_face(cell, face)
        if vector_to_face[-1] > 0:
            passmat = Operator.get_swaped_face_reference_frame_transformation_matrix(cell, face)
        else:
            passmat = face.reference_frame_transformation_matrix
        return passmat

    def get_line_from_indices(self, i: int, j: int, unknown: Unknown) -> int:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Returns :
        ================================================================================================================
        - 
        """
        for line, index in enumerate(unknown.indices):
            if index[0] == i and index[1] == j:
                return line
            if unknown.symmetric_gradient and index[0] == j and index[1] == i:
                return line
        raise ValueError("ATtention")
