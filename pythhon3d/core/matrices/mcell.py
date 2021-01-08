import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.unknown import Unknown
from core.integration import Integration
from bases.basis import Basis


class Mcell:
    def __init__(
        self,
        cell_vertices_connectivity_matrix: Mat,
        cell: Cell,
        cell_basis_l: Basis,
        cell_basis_k: Basis,
        cell_basis_k1: Basis,
        unknown: Unknown,
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
        self.mat_T_phi_k_phi_k_inv = np.linalg.inv(mat_T_phi_k_phi_k)
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        self.mat_T_phi_k_grad_phi_l_list = []
        for j in range(unknown.problem_dimension):
            mat_T_phi_k_grad_phi_l = Integration.get_cell_advection_matrix_in_cell(cell, cell_basis_k, cell_basis_l, j)
            self.mat_T_phi_k_grad_phi_l_list.append(mat_T_phi_k_grad_phi_l)
        # ==============================================================================================================
        # RECONSTRUCTION OPERATOR
        # ==============================================================================================================
        # left-hand side
        # --------------------------------------------------------------------------------------------------------------
        self.mat_T_grad_phi_k1_grad_phi_k1_inv_list = []
        for j in range(unknown.problem_dimension):
            mat_T_grad_phi_k1_grad_phi_k1_j = Integration.get_cell_stiffness_matrix_in_cell(
                cell, cell_basis_k1, cell_basis_k1, j
            )
            self.mat_T_grad_phi_k1_grad_phi_k1_inv_list.append(mat_T_grad_phi_k1_grad_phi_k1_j)
        # --------------------------------------------------------------------------------------------------------------
        # right-hand side
        # --------------------------------------------------------------------------------------------------------------
        self.mat_T_grad_phi_k1_phi_l_list = []
        for j in range(unknown.problem_dimension):
            mat_T_phi_l_grad_phi_k1_j = Integration.get_cell_advection_matrix_in_cell(
                cell, cell_basis_l, cell_basis_k1, j
            )
            mat_T_grad_phi_k1_phi_l_j = mat_T_phi_l_grad_phi_k1_j.T
            self.mat_T_grad_phi_k1_phi_l_list.append(mat_T_grad_phi_k1_phi_l_j)
        # ==============================================================================================================
        # STABILIZATION OPERATOR
        # ==============================================================================================================
        mat_T_phi_k_phi_k1 = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_k, cell_basis_k1)
        self.proj_T_T_k1_k = mat_T_phi_k_phi_k_inv @ mat_T_phi_k_phi_k1
        # ==============================================================================================================
        # PROJECTION OPERATOR
        # ==============================================================================================================
        mat_T_phi_l_phi_l = Integration.get_cell_mass_matrix_in_cell(cell, cell_basis_l, cell_basis_l)
        self.mat_T_phi_l_phi_l_inv = np.linalg.inv(mat_T_phi_l_phi_l)
