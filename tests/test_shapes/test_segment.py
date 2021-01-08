from tests import context
import pytest
import numpy as np
from scipy import integrate
from shapes.segment import Segment


def test_segment_centroid():
    segment = np.array([[0.0], [1.0]])
    polynomial_order = 1
    s = Segment(segment, polynomial_order)
    expected_centroid = np.array([0.5])
    assert (np.abs(s.centroid - expected_centroid) < np.full((1,), 1.0e-9)).all() and s.centroid.shape == (1,)


def test_segment_volume():
    segment = np.array([[0.0], [1.0]])
    polynomial_order = 1
    s = Segment(segment, polynomial_order)
    expected_volume = 1.0
    assert np.abs(s.volume - expected_volume) < 1.0e-9


polynomial_orders = [1, 2, 3, 4, 5, 6, 7, 8]

functions_segment = [
    (lambda x: 2.0 * x ** 1 + 3.0),
    (lambda x: 1.0 * x ** 2 + 6.0 * x ** 1 - 2.0),
    (lambda x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x ** 1 - 1.0),
    (lambda x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x ** 1 + 9.0),
    (lambda x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x ** 1 - 6.0),
    (lambda x: 2.0 * x ** 6 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x ** 1 - 9.0),
    (lambda x: 2.0 * x ** 7 - 1.0 * x ** 5 + 1.0 * x ** 6 - 3.0 * x ** 2 + 1.0 * x ** 1 - 4.0),
    (lambda x: 2.0 * x ** 8 - 1.0 * x ** 3 + 1.0 * x ** 7 - 3.0 * x ** 2 + 1.0 * x ** 1 - 2.0),
    # (lambda x: 2.0 * x),
]

test_data = []
for k in polynomial_orders:
    test_data.append((k, integrate.quad(functions_segment[k - 1], 0.0, 1.2)[0]))


@pytest.mark.parametrize("k, expected", test_data)
def test_segment_quadrature(k, expected):
    segment = np.array([[0.0], [1.2]])
    s = Segment(segment, k)
    quadrature_points, quadrature_weights = s.quadrature_points, s.quadrature_weights
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_segment[k - 1](quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(s.quadrature_points, s.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-10


# integral_bounds = [
#     list(map(lambda x: x ** 2 + 3.0 * x, [-0.2, 1.6])),
#     list(map(lambda x: (1.0 / 3.0) * x ** 3 + (6.0 / 2.0) * x ** 2 - 2.0 * x, [-0.2, 1.6])),
#     list(map(lambda x: x ** 4 - (4.0 / 3.0) * x ** 3 + (5.0 / 2.0) * x ** 2 - x, [-0.2, 1.6])),
#     list(
#         map(
#             lambda x: (7.0 / 5.0) * x ** 5 - x ** 4 + (5.0 / 3.0) * x ** 3 - (1.0 / 2.0) * x ** 2 + 9.0 * x, [-0.2, 1.6]
#         )
#     ),
#     list(
#         map(
#             lambda x: (2.0 / 6.0) * x ** 6
#             - (1.0 / 5.0) * x ** 5
#             + (1.0 / 4.0) * x ** 4
#             - x ** 3
#             + (1.0 / 2.0) * x ** 2
#             - 6.0 * x,
#             [-0.2, 1.6],
#         )
#     ),
# ]

# test_data = [
#     (1, integral_bounds[0][1] - integral_bounds[0][0]),
#     (2, integral_bounds[1][1] - integral_bounds[1][0]),
#     (3, integral_bounds[2][1] - integral_bounds[2][0]),
#     (4, integral_bounds[3][1] - integral_bounds[3][0]),
#     (5, integral_bounds[4][1] - integral_bounds[4][0]),
# ]


# @pytest.mark.parametrize("k, expected", test_data)
# def test_segment_quadrature(k, expected):
#     segment = np.array([[-0.2], [1.6]])
#     s = Segment(segment, k)
#     numerical_integral = np.sum([w * functions[k - 1](x) for x, w in zip(s.quadrature_nodes, s.quadrature_weights)])
#     assert np.abs(numerical_integral - expected) < 1.0e-9
