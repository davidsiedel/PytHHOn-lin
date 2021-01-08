import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from shapes.domain import Domain
from core.face import Face
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis


class Mface1d:
    def __init__(self, face_vertices_connectivity_matrix: Mat, face: Face, face_basis_k: Basis):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Mface class 
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - 
        """
        # --------------------------------------------------------------------------------------------------------------
        mat_F_psi_k_psi_k = Integration.get_face_mass_matrix_in_face(
            face, face_basis_k, face_basis_k, face.reference_frame_transformation_matrix
        )
        mat_F_psi_k_psi_k_inv = np.linalg.inv(mat_F_psi_k_psi_k)
        # --------------------------------------------------------------------------------------------------------------

