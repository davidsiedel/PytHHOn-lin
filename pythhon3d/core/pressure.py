import numpy as np
from numpy import ndarray as Mat
from typing import List
from typing import Callable

from core.cell import Cell
from core.face import Face
from bases.basis import Basis
from core.integration import Integration
from core.unknown import Unknown


class Pressure:
    def __init__(
        self,
        face: Face,
        face_basis: Basis,
        face_reference_frame_transformation_matrix: Mat,
        unknown: Unknown,
        pressure: List[Callable] = None,
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
        if pressure is None:
            pressure_vector = np.zeros((unknown.field_dimension * face_basis.basis_dimension,))
        else:
            pressure_vector = np.zeros((unknown.field_dimension * face_basis.basis_dimension,))
            for i, pressure_component in enumerate(pressure):
                if pressure_component is None:
                    pressure_vector_component = np.zeros((face_basis.basis_dimension,))
                else:
                    # face: Face, face_basis: Basis, face_reference_frame_transformation_matrix: Mat, pressure: Callable,
                    pressure_vector_component = Integration.get_face_pressure_vector_in_face(
                        face, face_basis, face_reference_frame_transformation_matrix, pressure_component
                    )
                pressure_vector[
                    i * face_basis.basis_dimension : (i + 1) * face_basis.basis_dimension
                ] += pressure_vector_component
        self.pressure_vector = pressure_vector

    def check_pressure_consistency(self, pressure: List[Callable], unknown: Unknown):
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
        if not len(pressure) == unknown.field_dimension:
            raise ValueError("Attention")
