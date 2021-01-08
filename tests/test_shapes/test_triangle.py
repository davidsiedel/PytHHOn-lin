from tests import context
import pytest
import numpy as np
from scipy import integrate
from shapes.triangle import Triangle


def test_triangle_centroid():
    triangle = np.array([[1.0, 2.0], [1.6, 2.0], [1.2, 2.8]])
    polynomial_order = 1
    t = Triangle(triangle, polynomial_order)
    expected_centroid = np.array([1.2666666666666666, 2.2666666666666666])
    assert (np.abs(t.centroid - expected_centroid) < np.full((2,), 1.0e-9)).all() and t.centroid.shape == (2,)


def test_triangle_volume():
    triangle = np.array([[1.0, 2.0], [1.6, 2.0], [1.2, 2.8]])
    polynomial_order = 1
    t = Triangle(triangle, polynomial_order)
    expected_volume = 0.24
    assert t.volume == expected_volume


def test_triangle_diameter():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    polynomial_order = 1
    t = Triangle(triangle, polynomial_order)
    expected_diameter = np.sqrt(2.0)
    assert t.diameter == expected_diameter


polynomial_orders = [1, 2, 3, 4, 5, 6, 7, 8]

functions_triangle = [
    (lambda y, x: 2.0 * x ** 1 + 3.0 * y ** 1 + 7.0),
    (lambda y, x: 1.0 * x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0),
    (lambda y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3),
    (lambda y, x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0 + 5.0 * y ** 4),
    (lambda y, x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0 + 6.0 * y ** 5),
    (lambda y, x: 2.0 * x ** 3 * y ** 3 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0 + 6.0 * y ** 6),
    (lambda y, x: 2.0 * x ** 3 * y ** 4 + 1.0 * x ** 3 - 3.0 * y ** 7 + 1.0 * x - 6.0 + 6.0 * y ** 7),
    # (lambda y, x: 2.0 * x ** 4 * y ** 4 + 1.0 * x ** 3 - 3.0 * y ** 7 + 1.0 * x - 6.0 + 6.0 * y ** 8),
    (
        lambda y, x: 2.0 * (x ** 4) * (y ** 4)
        + 1.0 * x ** 3
        - 3.0 * y ** 7
        + 1.0 * x
        - 6.0
        + 6.0 * y ** 8
        - 123.0 * x ** 8
        + 530.0 * (x ** 3) * (y ** 4)
    ),
]

test_data = []
for k in polynomial_orders:
    test_data.append(
        (k, integrate.dblquad(functions_triangle[k - 1], 0.0, 1.2, lambda x: 0.0, lambda x: 1 - (x / 1.2))[0])
    )


@pytest.mark.parametrize("k, expected", test_data)
def test_triangle_quadrature(k, expected):
    triangle = np.array([[0.0, 0.0], [1.2, 0.0], [0.0, 1.0]])
    t = Triangle(triangle, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_triangle[k - 1](quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(t.quadrature_points, t.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-10
