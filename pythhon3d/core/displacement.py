import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.face import Face
from bases.basis import Basis
from core.integration import Integration
from core.unknown import Unknown


class Displacement:
    def __init__(
        self,
        face: Face,
        face_basis: Basis,
        face_reference_frame_transformation_matrix: Mat,
        displacement: List[Callable],
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
        displacement_vector = Integration.get_face_pressure_vector_in_face(
            face, face_basis, face_reference_frame_transformation_matrix, displacement
        )
        self.displacement_vector = displacement_vector

    def check_displacement_consistency(self, displacement: List[Callable], unknown: Unknown):
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
        if not len(displacement) == unknown.field_dimension:
            raise ValueError("Attention")

    @staticmethod
    def get_face_displacement_vector_in_face(
        face: Face, face_basis: Basis, face_reference_frame_transformation_matrix: Mat, pressure: Callable,
    ) -> Mat:
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
        v_f = face.diameter
        x_f = face.centroid
        # --------------------------------------------------------------------------------------------------------------
        face_pressure_vector_in_face = np.zeros((face_basis.basis_dimension,))
        # --------------------------------------------------------------------------------------------------------------
        x_f_in_face = Face.get_points_in_face_reference_frame(face.centroid, face_reference_frame_transformation_matrix)
        face_quadrature_nodes_in_face = Face.get_points_in_face_reference_frame(
            face.quadrature_nodes, face_reference_frame_transformation_matrix
        )
        for x_Q_f_in_face, w_Q_f in zip(face_quadrature_nodes_in_face, face.quadrature_weights):
            # ----------------------------------------------------------------------------------------------------------
            psi_vector = face_basis.get_phi_vector(x_Q_f_in_face, x_f_in_face, v_f)
            v = w_Q_f * psi_vector * pressure(x_Q_f_in_face)
            face_pressure_vector_in_face += v
        return face_pressure_vector_in_face
