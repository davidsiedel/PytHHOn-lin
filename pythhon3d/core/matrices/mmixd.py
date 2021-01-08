import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.face import Face
from core.matrices.mcell import Mcell
from core.matrices.mface import Mface
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis


class Mmixd:
    def __init__(
        self,
        cell_faces_connectivity_matrix: Mat,
        mcell: Mcell,
        mfaces: List[Mface],
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        unknown: Unknown,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Mbox class 
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - 
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - 
        """
        # ==============================================================================================================
        # GRADIENT OPERATOR / STABILIZATION OPERATOR
        # ==============================================================================================================
        self.mat_F_phi_k_phi_l_list = []
        self.mat_F_phi_k_psi_k_list = []
        for mface in mfaces:
            mat_F_phi_k_phi_l = Integration.get_cell_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k, cell_basis_l
            )
            self.mat_F_phi_k_phi_l_list.append(mat_F_phi_k_phi_l_list)
            # ----------------------------------------------------------------------------------------------------------
            mat_F_phi_k_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k, face_basis_k, mface.face.reference_frame_transformation_matrix
            )
            self.mat_F_phi_k_psi_k_list.append(mat_F_phi_k_psi_k)
        # ==============================================================================================================
        # RECONSTRUCTION OPERATOR
        # ==============================================================================================================
        self.mat_F_grad_phi_k1_psi_k_list = []
        self.mat_F_grad_phi_k1_phi_l_list = []
        for mface in mfaces:
            mat_F_grad_phi_k1_psi_k_j_list = []
            mat_F_grad_phi_k1_phi_l_j_list = []
            for j in range(len(unknown.problem_dimension)):
                # ------------------------------------------------------------------------------------------------------
                mat_F_grad_phi_k1_psi_k_j = Integration.get_hybrid_advection_matrix_in_face(
                    mcell.cell,
                    mface.face,
                    cell_basis_k1,
                    face_basis_k,
                    mface.face.reference_frame_transformation_matrix,
                    j,
                )
                mat_F_grad_phi_k1_psi_k_j_list.append(mat_F_grad_phi_k1_psi_k_j)
                # ------------------------------------------------------------------------------------------------------
                mat_F_phi_l_grad_phi_k1_j = Integration.get_cell_advection_matrix_in_face(
                    mcell.cell, mface.face, cell_basis_l, cell_basis_k1, j
                )
                # ------------------------------------------------------------------------------------------------------
                mat_F_grad_phi_k1_phi_l_j = mat_F_phi_l_grad_phi_k1_j.T
                mat_F_grad_phi_k1_phi_l_j_list.append(mat_F_grad_phi_k1_phi_l_j)
            # ----------------------------------------------------------------------------------------------------------
            self.mat_F_grad_phi_k1_psi_k_list.append(mat_F_grad_phi_k1_psi_k_j_list)
            self.mat_F_grad_phi_k1_phi_l_list.append(mat_F_grad_phi_k1_phi_l_j_list)
        # ==============================================================================================================
        # STABILIZATION OPERATOR
        # ==============================================================================================================
        self.mat_F_psi_k_phi_k1_list = []
        for mface in mfaces:
            mat_F_phi_k1_psi_k = Integration.get_hybrid_mass_matrix_in_face(
                mcell.cell, mface.face, cell_basis_k1, face_basis_k, mface.face.reference_frame_transformation_matrix
            )
            self.mat_F_psi_k_phi_k1_list.append(mat_F_phi_k1_psi_k)
