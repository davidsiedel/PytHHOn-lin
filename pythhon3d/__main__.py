import argparse
import numpy as np

from parsers.geof_parser import parse_geof_file as parse_mesh
from parsers.element_types import C_cf_ref
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from core.face import Face
from core.cell import Cell
from core.operators.operator import Operator
from core.operators.hdg import HDG

from pythhon3d import build, solve

import pickle
import sys, os
from pathlib import Path

if __name__ == "__main__":
    current_folder = Path(os.path.dirname(os.path.abspath(__file__)))
    source = current_folder.parent
    # ------------------------------------------------------------------------------------------------------------------
    bld_file_path = os.path.join(source, "_inputs/bld.bin")
    with open(bld_file_path, "rb") as bld_file:
        bld_obj = pickle.load(bld_file)
    (mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type) = bld_obj
    # ------------------------------------------------------------------------------------------------------------------
    (
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets,
        cell_basis,
        face_basis,
        unknown,
    ) = build(mesh_file, field_dimension, face_polynomial_order, cell_polynomial_order, operator_type)
    # ------------------------------------------------------------------------------------------------------------------
    d = unknown.problem_dimension
    tangent_matrices = [
        np.eye(d ** 2) for i in range(len(cells))
    ]  # ------------------------------------------------------------------------------------------------------------------
    cpt_file_path = os.path.join(source, "_inputs/cpt.bin")
    with open(cpt_file_path, "rb") as cpt_file:
        cpt_obj = pickle.load(cpt_file)
    (stabilization_parameter, load, boundary_conditions) = cpt_obj
    # ------------------------------------------------------------------------------------------------------------------
    (vertices, unknowns_at_vertices), (quadrature_points, unknowns_at_quadrature_points) = solve(
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets,
        cell_basis,
        face_basis,
        unknown,
        tangent_matrices,
        stabilization_parameter,
        boundary_conditions,
        load,
    )
    # ------------------------------------------------------------------------------------------------------------------
    out_file_path = os.path.join(source, "_outputs/out.bin")
    with open(out_file_path, "rb") as out_file:
        pickle.dump(((vertices, unknowns_at_vertices), (quadrature_points, unknowns_at_quadrature_points)), out_file)

