import numpy as np

from parsers.element_types import C_cf_ref


def parse_geof_file(geof_file_path):
    """
    ====================================================================================================================
    Function :
    ====================================================================================================================
    Reads a .geof mesh file, and extracts needed data about the structure to build the HHO framework
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    - geof_file_path : the absolute path to the mesh file to read
    ====================================================================================================================
    Returns :
    ====================================================================================================================
    - problem_dimension : the integer that defines the dimension of the euclidian space.
    - vertices : the matrix containing all coordinates of the vertices as vectors. Each row corresponds to a vertex in
    the mesh, and the number of colums is the number of vertices in the mesh.
    - cells_vertices_connectivity_matrix : the connectivity matrix, that links a cell to vertices indices in the 
    vertices matrix. rows in cells_vertices_connectivity_matrix correspon to indices of cells in the mesh (i.e. the
    equivalent of an element in the FE methdod using Lagrange elements), and columns in
    cells_vertices_connectivity_matrix are the vertices indices to which the cell is attached.
    - faces_vertices_connectivity_matrix : the connectivity matrix, that links a face to vertices indices in the 
    vertices matrix, with same strcuture as cells_vertices_connectivity_matrix.
    - cells_faces_connectivity_matrix : the connectivity matrix, that links a cell to faces indices. A face index corresponds to a row index in faces_vertices_connectivity_matrix.
    - cells_connectivity_matrix : 
    - boundaries_init : a dictionary with correspondance between a vertices set name, denoted with a string, and a list of 
    vertex indices, containing the indices of all vertices connected to that vertices set.
    - boundaries : a dictionary with correspondance between a faces set name, denoted with a string, and a list of 
    face indices, containing the indices of all faces connected to that vertices set.
    """
    with open(geof_file_path, "r") as geof_file:
        c = geof_file.readlines()
        i_line = 2
        # --------------------------------------------------------------------------------------------------------------
        # reading the dimension of the nodes matrix.
        # --------------------------------------------------------------------------------------------------------------
        N_size_rows = int(c[i_line].rstrip().split(" ")[0])
        N_size_cols = int(c[i_line].rstrip().split(" ")[1])
        problem_dimension = N_size_cols
        # --------------------------------------------------------------------------------------------------------------
        # skipping the line to the first nodes.
        # --------------------------------------------------------------------------------------------------------------
        i_line += 1
        vertices = np.zeros((N_size_rows, N_size_cols))
        vertices_weights = np.zeros((N_size_rows,), dtype=np.float32)
        i_ver = 0
        for i in range(i_line, i_line + N_size_rows):
            vertex = [float(c[i].split(" ")[j]) for j in range(1, N_size_cols + 1)]
            vertex = np.array(vertex, dtype=np.float64)
            vertices[i_ver, :] += vertex
            i_ver += 1
            # vertices.append(vertex)
        # --------------------------------------------------------------------------------------------------------------
        i_line += N_size_rows + 1
        # --------------------------------------------------------------------------------------------------------------
        # reading the dimension of the C_nc matrix.
        # --------------------------------------------------------------------------------------------------------------
        N_cells = int(c[i_line].rstrip())
        # --------------------------------------------------------------------------------------------------------------
        # EXTRACTING THE NSETS MATRICES
        # --------------------------------------------------------------------------------------------------------------
        # skipping the extraction of the cells connectivity matrix to extract
        # directly the boundaries_init matrices. By doing so, it is easier when defining
        # faces afterward to check whether they belong to a Nset or not. hence,
        # one defines a temporary line increment for the reading of boundaries_init
        # related lines.
        # --------------------------------------------------------------------------------------------------------------
        i_line_boundaries_init = i_line + N_cells + 2
        boundaries_init = {}
        for i in range(i_line_boundaries_init, len(c)):
            if "**liset" in c[i] or "**elset" in c[i]:
                break
            else:
                if "**nset" in c[i]:
                    boundary_name = (c[i]).split(" ")[1].rstrip()
                    boundary_nodes = []
                else:
                    if not "***" in c[i]:
                        if c[i].split(" ")[0] == "":
                            start_point = 1
                        else:
                            start_point = 0
                        nodes = [int(item.replace("\n", "")) - 1 for item in c[i].split(" ")[start_point:]]
                        # print(c[i])
                        # nodes = [
                        #     int(item.replace("\n", "").replace(" ", "")) - 1 for item in c[i].split(" ")[start_point:]
                        # ]
                        boundary_nodes += nodes
                    else:
                        boundaries_init[boundary_name] = boundary_nodes
                        break
            boundaries_init[boundary_name] = boundary_nodes
        # --------------------------------------------------------------------------------------------------------------
        # EXTRACTING THE CELLS CONNECTIVITY MATRIX
        # --------------------------------------------------------------------------------------------------------------
        # getting back to the line where the connectivity between cells and
        # nodes is defined.
        # --------------------------------------------------------------------------------------------------------------
        i_line += 1
        # --------------------------------------------------------------------------------------------------------------
        # checking weather all elements are of the same nature, i.e. if the mesh
        # contains either c2d3 and c2d4 elements for instance. If so, the cells
        # connectivity matrix does not have a homogeneous number of columns.
        # --------------------------------------------------------------------------------------------------------------
        cells_types = []
        cells_connectivity_matrix = []
        cells_vertices_connectivity_matrix = []
        for i in range(i_line, i_line + N_cells):
            element_type = str(c[i].split(" ")[1])
            cells_types.append(element_type)
            cells_connectivity_matrix.append(C_cf_ref[element_type])
            cell_vertices = [int(c[i].split(" ")[j]) - 1 for j in range(2, len(c[i].split(" ")))]
            cells_vertices_connectivity_matrix.append(cell_vertices)
            cell_vertices_ones = np.ones((len(cell_vertices),), dtype=np.float32)
            vertices_weights[cell_vertices] += cell_vertices_ones
        # --------------------------------------------------------------------------------------------------------------
        # Initializing
        # - the faces connectivity matrix C_nf.
        # - the cell-face connectivity matrix C_cf.
        # - a tags list that stores a serial ID for each face, in order not to
        #  add the same face twice in the face connectivity matrix.
        # - a flags vector (with size the number fo faces in the mesh) to store
        # the boundaries_init each face belongs to.
        # - a weights vector with the weight of each node (the number of cells
        # it belongs to)
        # --------------------------------------------------------------------------------------------------------------
        cell_types = []
        for cell_type in cells_types:
            if not cell_type in cell_types:
                cell_types.append(cell_type)

        # weights = [0 for i in range(len(vertices))]
        weights = np.zeros((len(vertices),), dtype=int)
        tags = []
        flags = []
        cells_faces_connectivity_matrix = []
        faces_vertices_connectivity_matrix = []
        # --------------------------------------------------------------------------------------------------------------
        # For each cell :
        # --------------------------------------------------------------------------------------------------------------
        for i in range(len(cells_vertices_connectivity_matrix)):
            for vertex_index in cells_vertices_connectivity_matrix[i]:
                weights[vertex_index] += 1
            cell_face_connectivity_matrix = []
            cell_type = cells_types[i]
            # ----------------------------------------------------------------------------------------------------------
            # For each face in the ith cell :
            # ----------------------------------------------------------------------------------------------------------
            for j in range(len(C_cf_ref[cell_type])):
                # ------------------------------------------------------------------------------------------------------
                # Extracting the face connectivity matrix.
                # ------------------------------------------------------------------------------------------------------
                face_vertices_connectivity_matrix = []
                for k in C_cf_ref[cell_type][j]:
                    face_vertices_connectivity_matrix.append(cells_vertices_connectivity_matrix[i][k])
                tag = "".join([str(item) for item in np.sort(face_vertices_connectivity_matrix)])
                # ------------------------------------------------------------------------------------------------------
                # If the face is not stored yet :
                # ------------------------------------------------------------------------------------------------------
                if not tag in tags:
                    # --------------------------------------------------------------------------------------------------
                    # Store it in the face connectivity matrix.
                    # --------------------------------------------------------------------------------------------------
                    flags_local = []
                    faces_vertices_connectivity_matrix.append(face_vertices_connectivity_matrix)
                    tags.append(tag)
                    cell_face_connectivity_matrix.append(len(tags) - 1)
                    # --------------------------------------------------------------------------------------------------
                    # For all boundaries_init, check whether the face belongs to it or
                    # not. If it belongs to any Nset, append flag with the name
                    # of the Nset, and append "NONE" otherwise.
                    # --------------------------------------------------------------------------------------------------
                    for key in boundaries_init.keys():
                        count = 0
                        for vertex_index in face_vertices_connectivity_matrix:
                            if vertex_index in boundaries_init[key]:
                                count += 1
                        if count == len(face_vertices_connectivity_matrix):
                            flags_local.append(key)
                    if not flags_local:
                        flags_local = None
                    flags.append(flags_local)
                # ------------------------------------------------------------------------------------------------------
                # If the face is already stored :
                # ------------------------------------------------------------------------------------------------------
                else:
                    cell_face_connectivity_matrix.append(tags.index(tag))
            cells_faces_connectivity_matrix.append(cell_face_connectivity_matrix)
        # --------------------------------------------------------------------------------------------------------------
        # print("HEEERE : \n{}, {}".format(len(flags), len(faces_vertices_connectivity_matrix)))
        boundaries = {}
        for boundary_name in boundaries_init:
            boundaries[boundary_name] = []
        for i, local_flags in enumerate(flags):
            if not local_flags is None:
                for flag in local_flags:
                    boundaries[flag].append(i)
        # print("problem_dimension :\n {}\n".format(problem_dimension))
        # print("vertices :\n {}\n".format(vertices))
        # print("cells_vertices_connectivity_matrix :\n {}\n".format(cells_vertices_connectivity_matrix))
        # print("faces_vertices_connectivity_matrix :\n {}\n".format(faces_vertices_connectivity_matrix))
        # print("cells_faces_connectivity_matrix :\n {}\n".format(cells_faces_connectivity_matrix))
        # print("boundaries_init :\n {}\n".format(boundaries_init))
        # print("flags :\n {}\n".format(flags))
        # print("boundaries :\n {}\n".format(boundaries))
        # print(vertices_weights)
        return (
            problem_dimension,
            vertices,
            vertices_weights,
            cells_vertices_connectivity_matrix,
            faces_vertices_connectivity_matrix,
            cells_faces_connectivity_matrix,
            cells_connectivity_matrix,
            # boundaries_init,
            boundaries,
        )
