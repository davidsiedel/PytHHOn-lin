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

    # @staticmethod
    # def get_segment_quadrature(vertices, volume, quadrature_order):
    #     """
    #     ================================================================================================================
    #     Method :
    #     ================================================================================================================
    #     Computes the quadrature points and weights for a segment in the euclidian space
    #     ================================================================================================================
    #     Parameters :
    #     ================================================================================================================
    #     - vertices : the vertices of the segment
    #     - volume : the length of the segment
    #     - quadrature_order : the quadrature order, that is defined by the order of polynomials to integrate
    #     ================================================================================================================
    #     Returns :
    #     ================================================================================================================
    #     - quadrature_points : the quadrature points of the given segment.
    #     - quadrature_weights : the quadrature points of the given segment.
    #     """
    #     if quadrature_order == 1:
    #         barycentric_coordinates = np.array([[0.5000000000000000, 0.5000000000000000]])
    #         quadrature_points = []
    #         for barycentric_coordinate in barycentric_coordinates:
    #             node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
    #             quadrature_points.append(node_Q)
    #         quadrature_points = np.array(quadrature_points)
    #         quadrature_weights = np.array([[volume]])
    #     if quadrature_order in [2, 3]:
    #         barycentric_coordinates = np.array(
    #             [[0.78867513459481290, 0.21132486540518713], [0.21132486540518713, 0.78867513459481290],]
    #         )
    #         quadrature_points = []
    #         for barycentric_coordinate in barycentric_coordinates:
    #             node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
    #             quadrature_points.append(node_Q)
    #         quadrature_points = np.array(quadrature_points)
    #         quadrature_weights = np.array([[volume * 0.50000000000000000], [volume * 0.50000000000000000]])
    #     if quadrature_order in [4, 5]:
    #         barycentric_coordinates = np.array(
    #             [
    #                 [0.11270166537925830, 0.88729833462074170],
    #                 [0.50000000000000000, 0.50000000000000000],
    #                 [0.88729833462074170, 0.11270166537925830],
    #             ]
    #         )
    #         quadrature_points = []
    #         for barycentric_coordinate in barycentric_coordinates:
    #             node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
    #             quadrature_points.append(node_Q)
    #         quadrature_points = np.array(quadrature_points)
    #         quadrature_weights = np.array(
    #             [[volume * 0.2777777777777778], [volume * 0.4444444444444444], [volume * 0.2777777777777778],]
    #         )
    #     return quadrature_points, quadrature_weights

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
        mid = (vertices[1] + vertices[0]) / 2.0
        if quadrature_order in [1, 2]:
            quad_ref = np.array([[-0.5773502691896257645092], [0.5773502691896257645092]])
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array([[1.00000000000000000], [1.00000000000000000]]) * (volume / 2.0)
        if quadrature_order in [3, 4]:
            quad_ref = np.array(
                [
                    [-0.861136311594052575224],
                    [-0.3399810435848562648027],
                    [0.3399810435848562648027],
                    [0.861136311594052575224],
                ]
            )
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array(
                [
                    [0.3478548451374538573731],
                    [0.6521451548625461426269],
                    [0.6521451548625461426269],
                    [0.3478548451374538573731],
                ]
            ) * (volume / 2.0)
        if quadrature_order in [5, 6]:
            quad_ref = np.array(
                [
                    [-0.9324695142031520278123],
                    [-0.661209386466264513661],
                    [-0.2386191860831969086305],
                    [0.238619186083196908631],
                    [0.661209386466264513661],
                    [0.9324695142031520278123],
                ]
            )
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array(
                [
                    [0.17132449237917034504],
                    [0.36076157304813860757],
                    [0.46791393457269104739],
                    [0.46791393457269104739],
                    [0.36076157304813860757],
                    [0.17132449237917034504],
                ]
            ) * (volume / 2.0)
        if quadrature_order in [7, 8]:
            quad_ref = np.array(
                [
                    [-0.9602898564975362316836],
                    [-0.7966664774136267395916],
                    [-0.5255324099163289858177],
                    [-0.1834346424956498049395],
                    [0.1834346424956498049395],
                    [0.5255324099163289858177],
                    [0.7966664774136267395916],
                    [0.9602898564975362316836],
                ]
            )
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array(
                [
                    [0.1012285362903762591525],
                    [0.2223810344533744705444],
                    [0.313706645877887287338],
                    [0.3626837833783619829652],
                    [0.3626837833783619829652],
                    [0.313706645877887287338],
                    [0.222381034453374470544],
                    [0.1012285362903762591525],
                ]
            ) * (volume / 2.0)
        if quadrature_order in [9, 10]:
            quad_ref = np.array(
                [
                    [-0.97390652851717172007],
                    [-0.86506336668898451073],
                    [-0.67940956829902440623],
                    [-0.43339539412924719079],
                    [-0.14887433898163121088],
                    [0.148874338981631210884],
                    [0.433395394129247190799],
                    [0.679409568299024406234],
                    [0.865063366688984510732],
                    [0.973906528517171720078],
                ]
            )
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array(
                [
                    [0.0666713443086881375936],
                    [0.149451349150580593146],
                    [0.219086362515982043996],
                    [0.2692667193099963550912],
                    [0.2955242247147528701739],
                    [0.295524224714752870174],
                    [0.269266719309996355091],
                    [0.2190863625159820439955],
                    [0.1494513491505805931458],
                    [0.0666713443086881375936],
                ]
            ) * (volume / 2.0)
        if quadrature_order in [11, 12]:
            quad_ref = np.array(
                [
                    [-0.9815606342467192506906],
                    [-0.9041172563704748566785],
                    [-0.769902674194304687037],
                    [-0.5873179542866174472967],
                    [-0.3678314989981801937527],
                    [-0.1252334085114689154724],
                    [0.1252334085114689154724],
                    [0.3678314989981801937527],
                    [0.5873179542866174472967],
                    [0.7699026741943046870369],
                    [0.9041172563704748566785],
                    [0.9815606342467192506906],
                ]
            )
            quad_mid = np.full(quad_ref.shape, mid)
            quadrature_points = (volume / 2.0) * quad_ref + quad_mid
            quadrature_weights = np.array(
                [
                    [0.0471753363865118271946],
                    [0.1069393259953184309603],
                    [0.1600783285433462263347],
                    [0.2031674267230659217491],
                    [0.233492536538354808761],
                    [0.2491470458134027850006],
                    [0.2491470458134027850006],
                    [0.233492536538354808761],
                    [0.203167426723065921749],
                    [0.160078328543346226335],
                    [0.1069393259953184309603],
                    [0.0471753363865118271946],
                ]
            ) * (volume / 2.0)
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
        if quadrature_order == 6:
            barycentric_coordinates = np.array(
                [
                    [0.501426509658179, 0.249286745170910, 0.249286745170910],
                    [0.249286745170910, 0.501426509658179, 0.249286745170910],
                    [0.249286745170910, 0.249286745170910, 0.501426509658179],
                    #
                    [0.873821971016996, 0.063089014491502, 0.063089014491502],
                    [0.063089014491502, 0.873821971016996, 0.063089014491502],
                    [0.063089014491502, 0.063089014491502, 0.873821971016996],
                    #
                    [0.053145049844817, 0.310352451033784, 0.636502499121399],
                    [0.053145049844817, 0.636502499121399, 0.310352451033784],
                    [0.636502499121399, 0.053145049844817, 0.310352451033784],
                    [0.310352451033784, 0.053145049844817, 0.636502499121399],
                    [0.310352451033784, 0.636502499121399, 0.053145049844817],
                    [0.636502499121399, 0.310352451033784, 0.053145049844817],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.116786275726379 * volume],
                    [0.116786275726379 * volume],
                    [0.116786275726379 * volume],
                    #
                    [0.050844906370207 * volume],
                    [0.050844906370207 * volume],
                    [0.050844906370207 * volume],
                    #
                    [0.082851075618374 * volume],
                    [0.082851075618374 * volume],
                    [0.082851075618374 * volume],
                    [0.082851075618374 * volume],
                    [0.082851075618374 * volume],
                    [0.082851075618374 * volume],
                ]
            )
        if quadrature_order in [7, 8]:
            barycentric_coordinates = np.array(
                [
                    [0.333333333333333, 0.333333333333333, 0.333333333333333],
                    #
                    [0.081414823414554, 0.459292588292723, 0.459292588292723],
                    [0.459292588292723, 0.081414823414554, 0.459292588292723],
                    [0.459292588292723, 0.459292588292723, 0.081414823414554],
                    #
                    [0.658861384496480, 0.170569307751760, 0.170569307751760],
                    [0.170569307751760, 0.658861384496480, 0.170569307751760],
                    [0.170569307751760, 0.170569307751760, 0.658861384496480],
                    #
                    [0.898905543365938, 0.050547228317031, 0.050547228317031],
                    [0.050547228317031, 0.898905543365938, 0.050547228317031],
                    [0.050547228317031, 0.050547228317031, 0.898905543365938],
                    #
                    [0.008394777409958, 0.263112829634638, 0.728492392955404],
                    [0.008394777409958, 0.728492392955404, 0.263112829634638],
                    [0.263112829634638, 0.008394777409958, 0.728492392955404],
                    [0.728492392955404, 0.008394777409958, 0.263112829634638],
                    [0.263112829634638, 0.728492392955404, 0.008394777409958],
                    [0.728492392955404, 0.263112829634638, 0.008394777409958],
                ]
            )
            quadrature_points = []
            for barycentric_coordinate in barycentric_coordinates:
                node_Q = np.sum((vertices * np.array([barycentric_coordinate]).T), axis=0)
                quadrature_points.append(node_Q)
            quadrature_points = np.array(quadrature_points)
            quadrature_weights = np.array(
                [
                    [0.144315607677787 * volume],
                    #
                    [0.095091634267285 * volume],
                    [0.095091634267285 * volume],
                    [0.095091634267285 * volume],
                    #
                    [0.103217370534718 * volume],
                    [0.103217370534718 * volume],
                    [0.103217370534718 * volume],
                    #
                    [0.032458497623198 * volume],
                    [0.032458497623198 * volume],
                    [0.032458497623198 * volume],
                    #
                    [0.027230314174435 * volume],
                    [0.027230314174435 * volume],
                    [0.027230314174435 * volume],
                    [0.027230314174435 * volume],
                    [0.027230314174435 * volume],
                    [0.027230314174435 * volume],
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
