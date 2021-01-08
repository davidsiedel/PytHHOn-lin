import numpy as np


class DunavantRule:
    def __init__(self):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The DunavantRule class provides a frameworquadrature_order to compute the appropriate point to exactly evaluate integrals over
        a domain in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        None
        ================================================================================================================
        Attributes :
        ================================================================================================================
        None
        """

    @staticmethod
    def get_point_quadrature():
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the quadrature points and weights for a point in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        None
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrature_points : the quadrature points of the given point. In this case, no point is returned since the 
        quadrature of a point is the point per se
        - quadrature_weights : the quadrature points of the given point.
        """
        quadrature_points, quadrature_weights = np.array([[]]), np.array([[1.0]])
        return quadrature_points, quadrature_weights

    @staticmethod
    def get_segment_quadrature(vertices, volume, quadrature_order):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the quadrature points and weights for a segment in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the vertices of the segment
        - volume : the length of the segment
        - quadrature_order : the quadrature order, that is defined by the order of polynomials to integrate
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrature_points : the quadrature points of the given segment.
        - quadrature_weights : the quadrature points of the given segment.
        """
        if quadrature_order == 1:
            barycentric_coordinates = np.array([[0.5000000000000000, 0.5000000000000000]])
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array([[volume]])
        if quadrature_order in [2, 3]:
            barycentric_coordinates = np.array(
                [[0.78867513459481290, 0.21132486540518713], [0.21132486540518713, 0.78867513459481290],]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array([[volume * 0.50000000000000000], [volume * 0.50000000000000000]])
        if quadrature_order in [4, 5]:
            barycentric_coordinates = np.array(
                [
                    [0.11270166537925830, 0.88729833462074170],
                    [0.50000000000000000, 0.50000000000000000],
                    [0.88729833462074170, 0.11270166537925830],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [[volume * 0.2777777777777778], [volume * 0.4444444444444444], [volume * 0.2777777777777778],]
            )
        return quadrature_points, quadrature_weights

    @staticmethod
    def get_triangle_quadrature(vertices, volume, quadrature_order):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the quadrature points and weights for a triangle in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the vertices of the triangle
        - volume : the length of the triangle
        - quadrature_order : the quadrature order, that is defined by the order of polynomials to integrate
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrature_points : the quadrature points of the given triangle.
        - quadrature_weights : the quadrature points of the given triangle.
        """
        if quadrature_order == 1:
            barycentric_coordinates = np.array([[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]])
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array([[volume]])
        if quadrature_order == 2:
            barycentric_coordinates = np.array(
                [
                    [0.66666666666666666, 0.16666666666666666, 0.16666666666666666],
                    [0.16666666666666666, 0.66666666666666666, 0.16666666666666666],
                    [0.16666666666666666, 0.16666666666666666, 0.66666666666666666],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [[0.3333333333333333 * volume], [0.3333333333333333 * volume], [0.3333333333333333 * volume]]
            )
        if quadrature_order == 3:
            barycentric_coordinates = np.array(
                [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.6000000000000000, 0.2000000000000000, 0.2000000000000000],
                    [0.2000000000000000, 0.6000000000000000, 0.2000000000000000],
                    [0.2000000000000000, 0.2000000000000000, 0.6000000000000000],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [-0.5625000000000000 * volume],
                    [0.5208333333333333 * volume],
                    [0.5208333333333333 * volume],
                    [0.5208333333333333 * volume],
                ]
            )
        if quadrature_order == 4:
            barycentric_coordinates = np.array(
                [
                    [0.8168475729804590, 0.0915762135097710, 0.0915762135097710],
                    [0.0915762135097710, 0.8168475729804590, 0.0915762135097710],
                    [0.0915762135097710, 0.0915762135097710, 0.8168475729804590],
                    [0.1081030181680700, 0.4459484909159650, 0.4459484909159650],
                    [0.4459484909159650, 0.1081030181680700, 0.4459484909159650],
                    [0.4459484909159650, 0.4459484909159650, 0.1081030181680700],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.109951743655322 * volume],
                    [0.109951743655322 * volume],
                    [0.109951743655322 * volume],
                    [0.223381589678011 * volume],
                    [0.223381589678011 * volume],
                    [0.223381589678011 * volume],
                ]
            )
        if quadrature_order == 5:
            barycentric_coordinates = np.array(
                [
                    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                    [0.7974269853530870, 0.1012865073234560, 0.1012865073234560],
                    [0.1012865073234560, 0.7974269853530870, 0.1012865073234560],
                    [0.1012865073234560, 0.1012865073234560, 0.7974269853530870],
                    [0.0597158717897700, 0.4701420641051150, 0.4701420641051150],
                    [0.4701420641051150, 0.0597158717897700, 0.4701420641051150],
                    [0.4701420641051150, 0.4701420641051150, 0.0597158717897700],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.225000000000000 * volume],
                    [0.125939180544827 * volume],
                    [0.125939180544827 * volume],
                    [0.125939180544827 * volume],
                    [0.132394152788506 * volume],
                    [0.132394152788506 * volume],
                    [0.132394152788506 * volume],
                ]
            )
        return quadrature_points, quadrature_weights

    @staticmethod
    def get_quadrangle_quadrature(vertices, volume, quadrature_order):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the quadrature points and weights for a quadrangle in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the vertices of the quadrangle
        - volume : the length of the quadrangle
        - quadrature_order : the quadrature order, that is defined by the order of polynomials to integrate
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrature_points : the quadrature points of the given quadrangle.
        - quadrature_weights : the quadrature points of the given quadrangle.
        """
        if quadrature_order == 1:
            barycentric_coordinates = np.array(
                [[0.25000000000000000, 0.25000000000000000, 0.25000000000000000, 0.25000000000000000],]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array([[1.0000000000000000 * volume]])
        if quadrature_order in [2, 3]:
            barycentric_coordinates = np.array(
                [
                    [0.68301270189221940, 0.10566243270259354, 0.10566243270259354, 0.10566243270259354],
                    [0.10566243270259354, 0.68301270189221940, 0.10566243270259354, 0.10566243270259354],
                    [0.10566243270259354, 0.10566243270259354, 0.68301270189221940, 0.10566243270259354],
                    [0.10566243270259354, 0.10566243270259354, 0.10566243270259354, 0.68301270189221940],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                ]
            )
        if quadrature_order in [4, 5]:
            barycentric_coordinates = np.array(
                [
                    [0.25000000000000000, 0.25000000000000000, 0.25000000000000000, 0.25000000000000000],
                    #
                    [0.83094750193111260, 0.05635083268962915, 0.05635083268962915, 0.05635083268962915],
                    [0.05635083268962915, 0.83094750193111260, 0.05635083268962915, 0.05635083268962915],
                    [0.05635083268962915, 0.05635083268962915, 0.83094750193111260, 0.05635083268962915],
                    [0.05635083268962915, 0.05635083268962915, 0.05635083268962915, 0.83094750193111260],
                    #
                    [0.44364916731037085, 0.44364916731037085, 0.05635083268962915, 0.05635083268962915],
                    [0.05635083268962915, 0.44364916731037085, 0.44364916731037085, 0.05635083268962915],
                    [0.05635083268962915, 0.05635083268962915, 0.44364916731037085, 0.44364916731037085],
                    [0.44364916731037085, 0.05635083268962915, 0.05635083268962915, 0.44364916731037085],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.0771604938271605 / 1.0 * volume],
                    #
                    [0.1975308641975309 / 1.0 * volume],
                    [0.1975308641975309 / 1.0 * volume],
                    [0.1975308641975309 / 1.0 * volume],
                    [0.1975308641975309 / 1.0 * volume],
                    #
                    [0.1234567901234568 / 1.0 * volume],
                    [0.1234567901234568 / 1.0 * volume],
                    [0.1234567901234568 / 1.0 * volume],
                    [0.1234567901234568 / 1.0 * volume],
                    # [[volume * 0.2777777777777778], [volume * 0.4444444444444444], [volume * 0.2777777777777778],]
                ]
            )
        return quadrature_points, quadrature_weights

    @staticmethod
    def get_tetrahedron_quadrature(vertices, volume, quadrature_order):
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the quadrature points and weights for a tetrahedron in the euclidian space
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - vertices : the vertices of the tetrahedron
        - volume : the length of the tetrahedron
        - quadrature_order : the quadrature order, that is defined by the order of polynomials to integrate
        ================================================================================================================
        Returns :
        ================================================================================================================
        - quadrature_points : the quadrature points of the given tetrahedron.
        - quadrature_weights : the quadrature points of the given tetrahedron.
        """
        if quadrature_order == 1:
            barycentric_coordinates = np.array(
                [[0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000]]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array([[volume]])
        if quadrature_order == 2:
            barycentric_coordinates = np.array(
                [
                    [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
                    [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
                    [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
                    [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                    [0.2500000000000000 * volume],
                ]
            )
        if quadrature_order == 3:
            barycentric_coordinates = np.array(
                [
                    [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                    [0.5000000000000000, 0.1666666666666667, 0.1666666666666667, 0.1666666666666667],
                    [0.1666666666666667, 0.5000000000000000, 0.1666666666666667, 0.1666666666666667],
                    [0.1666666666666667, 0.1666666666666667, 0.5000000000000000, 0.1666666666666667],
                    [0.1666666666666667, 0.1666666666666667, 0.1666666666666667, 0.5000000000000000],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [-0.8000000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                    [0.4500000000000000 * volume],
                ]
            )
        # if quadrature_order == 4:
        #     barycentric_coordinates = np.array(
        #         [
        #             [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
        #             [0.7857142857142860, 0.0714285714285710, 0.0714285714285710, 0.0714285714285710],
        #             [0.0714285714285710, 0.7857142857142860, 0.0714285714285710, 0.0714285714285710],
        #             [0.0714285714285710, 0.0714285714285710, 0.7857142857142860, 0.0714285714285710],
        #             [0.0714285714285710, 0.0714285714285710, 0.0714285714285710, 0.7857142857142860],
        #             #
        #             [0.3994035761667990, 0.3994035761667990, 0.1005964238332010, 0.1005964238332010],
        #             [0.3994035761667990, 0.2500000000000000, 0.3994035761667990, 0.2500000000000000],
        #             [0.3994035761667990, 0.2500000000000000, 0.2500000000000000, 0.3994035761667990],
        #             [0.2500000000000000, 0.3994035761667990, 0.3994035761667990, 0.2500000000000000],
        #             [0.2500000000000000, 0.3994035761667990, 0.2500000000000000, 0.3994035761667990],
        #             [0.2500000000000000, 0.2500000000000000, 0.3994035761667990, 0.3994035761667990],
        #         ]
        #     )
        #     quadrature_points = []
        #     for barycentric_coordinate in barycentric_coordinates:
        #         node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
        #         quadrature_points.append(node_Q)
        #     quadrature_points = np.array(quadrature_points)
        #     quadrature_weights = np.array(
        #         [
        #             [-0.0131555555555556 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0076222222222222 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #             [0.0248888888888889 * volume],
        #         ]
        #     )
        if quadrature_order in [4, 5]:
            barycentric_coordinates = np.array(
                [
                    [0.2500000000000000, 0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                    #
                    [0.8160578438945539, 0.0919710780527230, 0.0919710780527230, 0.0919710780527230],
                    [0.0919710780527230, 0.8160578438945539, 0.0919710780527230, 0.0919710780527230],
                    [0.0919710780527230, 0.0919710780527230, 0.8160578438945539, 0.0919710780527230],
                    [0.0919710780527230, 0.0919710780527230, 0.0919710780527230, 0.8160578438945539],
                    #
                    [0.3604127443407402, 0.3197936278296299, 0.3197936278296299, 0.3197936278296299],
                    [0.3197936278296299, 0.3604127443407402, 0.3197936278296299, 0.3197936278296299],
                    [0.3197936278296299, 0.3197936278296299, 0.3604127443407402, 0.3197936278296299],
                    [0.3197936278296299, 0.3197936278296299, 0.3197936278296299, 0.3604127443407402],
                    #
                    [0.1381966011250105, 0.1381966011250105, 0.3618033988749895, 0.3618033988749895],
                    [0.1381966011250105, 0.3618033988749895, 0.1381966011250105, 0.3618033988749895],
                    [0.1381966011250105, 0.3618033988749895, 0.3618033988749895, 0.1381966011250105],
                    [0.3618033988749895, 0.1381966011250105, 0.1381966011250105, 0.3618033988749895],
                    [0.3618033988749895, 0.1381966011250105, 0.3618033988749895, 0.1381966011250105],
                    [0.3618033988749895, 0.3618033988749895, 0.1381966011250105, 0.1381966011250105],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.1185185185185185 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0719370837790186 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0690682072262724 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                    [0.0529100529100529 * volume],
                ]
            )
        return quadrature_points, quadrature_weights
