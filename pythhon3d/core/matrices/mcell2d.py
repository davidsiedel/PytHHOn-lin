import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis


class Mcell2D:
    def __init__(
        self,
        cell_vertices_connectivity_matrix: Mat,
        cell: Cell,
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Mcell class 
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
        # GRADIENT OPERATOR
        # ==============================================================================================================
        # left-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_phi_k_phi_k = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k)
        mat_T_phi_k_phi_k_inv = np.linalg.inv(mat_T_phi_k_phi_k)
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_phi_k_grad_phi_l_x = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, 0)
        mat_T_phi_k_grad_phi_l_y = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, 1)
        mat_T_phi_k_grad_phi_l_list = [mat_T_phi_k_grad_phi_l_x, mat_T_phi_k_grad_phi_l_y]
        # ==============================================================================================================
        # RECONSTRUCTION OPERATOR
        # ==============================================================================================================
        # left-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_grad_phi_k1_grad_phi_k1_x = Integration.get_cell_stiffness_matrix_in_cell(
            cell, cell_basis_k1, cell_basis_k1, 0
        )
        mat_T_grad_phi_k1_grad_phi_k1_y = Integration.get_cell_stiffness_matrix_in_cell(
            cell, cell_basis_k1, cell_basis_k1, 1
        )
        mat_T_grad_phi_k1_grad_phi_k1_inv_list = [mat_T_grad_phi_k1_grad_phi_k1_x, mat_T_grad_phi_k1_grad_phi_k1_y]
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        mat_T_phi_l_grad_phi_k1_x = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_l, cell_basis_k1, 0)
        mat_T_phi_l_grad_phi_k1_y = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_l, cell_basis_k1, 1)
        mat_T_grad_phi_k1_phi_l_list = [mat_T_phi_l_grad_phi_k1_x, mat_T_phi_l_grad_phi_k1_y]
        # ==============================================================================================================
        # STABILIZATION OPERATOR
        # ==============================================================================================================
        mat_T_phi_k_phi_k1 = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k1)
        proj_T_T_k1_k = mat_T_phi_k_phi_k_inv @ mat_T_phi_k_phi_k1
        # ==============================================================================================================
        # PROJECTION OPERATOR
        # ==============================================================================================================
        mat_T_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
        mat_T_phi_l_phi_l_inv = np.linalg.inv(mat_T_phi_l_phi_l)
