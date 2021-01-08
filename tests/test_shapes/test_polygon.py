from tests import context
import pytest
import numpy as np
from scipy import integrate
from shapes.polygon import Polygon


def test_polygon_centroid():
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    polynomial_order = 1
    p = Polygon(polygon, polynomial_order)
    expected_centroid = np.array([0.5, 0.5])
    assert (np.abs(p.centroid - expected_centroid) < np.full((2,), 1.0e-9)).all() and p.centroid.shape == (2,)


def test_square_centroid():
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    polynomial_order = 1
    p = Polygon(polygon, polynomial_order)
    expected_centroid = np.array([0.5, 0.5])
    assert (np.abs(p.centroid - expected_centroid) < np.full((2,), 1.0e-9)).all() and p.centroid.shape == (2,)


def test_polygon_volume():
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    polynomial_order = 1
    p = Polygon(polygon, polynomial_order)
    expected_volume = 1.0
    assert np.abs(p.volume - expected_volume) < 1.0e-9


polynomial_orders = [1, 2, 3, 4, 5]

functions_polygon = [
    (lambda y, x: 2.0 * x ** 1 + 3.0 * y ** 1 + 7.0),
    (lambda y, x: 1.0 * x ** 2 + 6.0 * x + 3.0 * y ** 2 - 2.0 * y + 2.0 * x * y - 2.0),
    (lambda y, x: 4.0 * x ** 3 - 4.0 * x ** 2 + 5.0 * x - 1.0 + 4.0 * y ** 3),
    (lambda y, x: 7.0 * x ** 4 - 4.0 * x ** 3 + 5.0 * x ** 2 - 1.0 * x + 9.0 + 5.0 * y ** 4),
    (lambda y, x: 2.0 * x ** 5 - 1.0 * x ** 4 + 1.0 * x ** 3 - 3.0 * x ** 2 + 1.0 * x - 6.0 + 6.0 * y ** 5),
]

test_data = []
for k in polynomial_orders:
    test_data.append((k, integrate.dblquad(functions_polygon[k - 1], 0.0, 1.0, lambda x: 0.0, lambda x: 1)[0]))


@pytest.mark.parametrize("k, expected", test_data)
def test_polygon_quadrature(k, expected):
    # vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    p = Polygon(polygon, k)
    # polygon = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # t = polygon(polygon, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_polygon[k - 1](quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(p.quadrature_points, p.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9


@pytest.mark.parametrize("k, expected", test_data)
def test_polygon_quadrature_2(k, expected):
    # vertices = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    p = Polygon(polygon, k)
    # polygon = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # t = polygon(polygon, k)
    numerical_integral = np.sum(
        [
            quadrature_weight * functions_polygon[k - 1](quadrature_point[1], quadrature_point[0])
            for quadrature_point, quadrature_weight in zip(p.quadrature_points, p.quadrature_weights)
        ]
    )
    assert np.abs(numerical_integral - expected) < 1.0e-9
