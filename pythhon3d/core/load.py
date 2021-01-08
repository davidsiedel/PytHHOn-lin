import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from bases.basis import Basis
from core.integration import Integration
from core.unknown import Unknown


class Load:
    def __init__(self, cell: Cell, cell_basis: Basis, unknown: Unknown, load: List[Callable] = None):
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
        if load is None:
            load_vector = np.zeros((unknown.field_dimension * cell_basis.basis_dimension,))
        else:
            load_vector = np.zeros((unknown.field_dimension * cell_basis.basis_dimension,))
            for i, load_component in enumerate(load):
                if load_component is None:
                    load_vector_component = np.zeros((cell_basis.basis_dimension,))
                else:
                    load_vector_component = Integration.get_cell_load_vector_in_cell(cell, cell_basis, load_component)
                load_vector[
                    i * cell_basis.basis_dimension : (i + 1) * cell_basis.basis_dimension
                ] -= load_vector_component
        self.load_vector = load_vector

    def check_load_consistency(self, load: List[Callable], unknown: Unknown):
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
        if not len(load) == unknown.field_dimension:
            raise ValueError("Attention")
