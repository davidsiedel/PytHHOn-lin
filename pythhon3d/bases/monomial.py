import numpy as np
from numpy import ndarray as Mat
from scipy.special import binom

from bases.basis import Basis


class ScaledMonomial(Basis):
    def __init__(
        self, polynomial_order: int, domain_dimension: int,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The ScaledMonomial class provides a framework and methods to build a (scaled) polynomial
        basis given a polynomial order and a spatial dimension.
        The ScaledMonomial class is then used in the Integration class to compute integration matrices through
        quadrature rules.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - polynomial_order : the polynomial order
        - domain_dimension : the spatial dimension for the support in which polynomials act
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - exponents : the exponents vector used to define exponents of monomials
        - conformal_gradients : the list of (conformal) gradient operators. Each ith-element of the list consists of
        the ith-directional gradient operator (i.e. the derivative operator with respect to the ith variable in the
        euclidian space)
        """
        super().__init__(polynomial_order, domain_dimension)
        # self.basis_dimension = int(binom(polynomial_order + domain_dimension, polynomial_order,))
        self.exponents = self.get_exponents(polynomial_order, domain_dimension,)
        self.conformal_gradients = []
        for dx in range(domain_dimension):
            grad_dx = self.get_gradient_operator(polynomial_order, domain_dimension, dx)
            self.conformal_gradients.append(grad_dx)

    def get_exponents(self, polynomial_order: int, domain_dimension: int) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the exponent vector for a d-variate polynom of order k to build the (scaled) monomial basis, where
        the exponent vector denotes the sum of mutli indexes alpha such that | alpha | < k.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - polynomial_order : the polynomial order
        - domain_dimension : the spatial dimension for the support in which polynomials act
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let consider the space of polynoms of :
        - order 1 in dimension 2 (I)
        - order 3 in dimension 1 (II)
        - order 1 in dimension 3 (III)
        the exponent vectors of such spaces are respectively :
        (I) | 0 0 |  (II) | 0 |  (III) | 0 0 0 |
            | 0 1 |       | 1 |        | 0 0 1 |
            | 1 0 |       | 2 |        | 0 1 0 |
            | 0 2 |       | 3 |        | 1 0 0 |
            | 1 1 |                             
            | 2 0 |                             
        and corresponding vectors of these bases are :
        (I) | x^0*y^0 |  (II) | x^0 |  (III) | x^0*y^0*z^0 |
            | x^0*y^1 |       | x^1 |        | x^0*y^0*z^1 |
            | x^1*y^0 |       | x^2 |        | x^0*y^1*z^0 |
            | x^0*y^2 |       | x^3 |        | x^1*y^0*z^0 |
            | x^1*y^1 |                               
            | x^2*y^0 |
        ================================================================================================================
        Returns :
        ================================================================================================================
        - exponents_matrix : the set of exponents vectors gathered in a matrix. The number of rows is the dimension of
        the polynomial basis, and the number of columns is the dimension of the euclidian space
        """
        if domain_dimension == 0:
            exponents_matrix = np.array([[0.0]])
        else:
            exponents_matrix = []
            if domain_dimension == 1:
                for s in range(polynomial_order + 1):
                    exponents_matrix.append([s])
            elif domain_dimension == 2:
                for polynomial_order in range(polynomial_order + 1):
                    for l in range(polynomial_order + 1):
                        m = polynomial_order - l
                        exponents_matrix.append(
                            [l, m,]
                        )
            elif domain_dimension == 3:
                for polynomial_order in range(polynomial_order + 1):
                    for l in range(polynomial_order + 1):
                        for m in range(polynomial_order + 1):
                            if not (l + m) > polynomial_order:
                                n = polynomial_order - (l + m)
                                exponents_matrix.append(
                                    [l, m, n,]
                                )
        return np.array(exponents_matrix, dtype=int,)

    def get_gradient_operator(self, polynomial_order: int, domain_dimension: int, dx: int) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the linear operator that acts on a vector in the monomial basis to return
        its derivative in the same basis.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - polynomial_order : the polynomial order
        - domain_dimension : the spatial dimension for the support in which polynomials act
        - dx : the derivative variable in the euclidian space
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let consider p(x) = a*x^0 + b*x^1 + c*x^2 in the monomial basis | 1, x, x^2 |
        of the space of polynomials of order 2 in dimension 1. Then :
        p'(x) = | 0 0 0 | @ | a | where | 0 0 0 | is the gradient operator.
                | 1 0 0 |   | b |       | 1 0 0 |
                | 0 2 0 |   | c |       | 0 2 0 |
        ================================================================================================================
        Returns :
        ================================================================================================================
        - gradient_operator : the gradient operator to compute the derivative of a monomial with respect to the dx
        variable
        """
        # --------------------------------------------------------------------------------------------------------------
        # Initializing the conformal gradient operator with zeros
        # --------------------------------------------------------------------------------------------------------------
        gradient_operator = np.zeros((self.basis_dimension, self.basis_dimension))
        # --------------------------------------------------------------------------------------------------------------
        # Copiying the exponent matrix
        # --------------------------------------------------------------------------------------------------------------
        d_exponents_matrix = np.copy(self.exponents)
        # --------------------------------------------------------------------------------------------------------------
        # Deriving a polynom, hence substracting a one-filled vector at the
        # derivative variable position
        # --------------------------------------------------------------------------------------------------------------
        d_exponents_matrix[:, dx,] = d_exponents_matrix[:, dx,] - np.ones(d_exponents_matrix[:, dx,].shape, dtype=int,)
        # --------------------------------------------------------------------------------------------------------------
        # Building the conformal gradient operator
        # --------------------------------------------------------------------------------------------------------------
        for i, coef in enumerate(self.exponents[:, dx,]):
            if not coef == 0:
                for j, exponents in enumerate(self.exponents):
                    if (exponents == d_exponents_matrix[i]).all():
                        gradient_operator[i, j] = coef
                        break
        return gradient_operator

    def get_phi_vector(self, point: Mat, centroid: Mat, diameter: float,) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the polynomial valued vector at a given point in the scaled monomial basis.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - point : a point in the euclidian space, denoted by a vector gathering its coordinates in the euclidian space
        - centroid : the centroid of the domain in the euclidian space on which the polynomial acts, denoted by a
        vector gathering its coordinates in the euclidian space
        - diameter : the diameter of the domain, a scalar value
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let the unite square T = [0,1]*[0,1] in R^2, A = (0.1, 0.4) a point in T, and
        B = (0.5, 0.5) the centroid of T. The diameter of T is 1.
        Then, the polynomial valued vector of A in the scaled monomial basis of order 2
        in T is :
        phi_vector = | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^0 |
                     | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^1 |
                     | ((0.1-0.5)/1.)^1*((0.4-0.5)/1.)^0 |
                     | ((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^2 |
                     | ((0.1-0.5)/1.)^1*((0.4-0.5)/1.)^1 |
                     | ((0.1-0.5)/1.)^2*((0.4-0.5)/1.)^0 |
        ================================================================================================================
        Returns :
        ================================================================================================================
        - phi_vector : the vector of size the dimension of the polynomial basis, whose values are the evaluation at the
        given point of each monomial
        """
        point_matrix = np.tile(point, (self.basis_dimension, 1,),)
        centroid_matrix = np.tile(centroid, (self.basis_dimension, 1,),)
        phi_matrix = ((point_matrix - centroid_matrix) / diameter) ** self.exponents
        phi_vector = np.prod(phi_matrix, axis=1,)
        return phi_vector

    def get_d_phi_vector(self, point: Mat, centroid: Mat, diameter: float, dx: int,) -> Mat:
        """
        ================================================================================================================
        Method :
        ================================================================================================================
        Computes the polynomial valued vector at a given point in the scaled monomial basis derivative with respect
        to a given direction.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - point : a point in the euclidian space, denoted by a vector gathering its coordinates in the euclidian space
        - centroid : the centroid of the domain in the euclidian space on which the polynomial acts, denoted by a
        vector gathering its coordinates in the euclidian space
        - diameter : the diameter of the domain, a scalar value
        - dx : the derivative variable in the euclidian space
        ================================================================================================================
        Exemple :
        ================================================================================================================
        Let the unite square T = [0,1]*[0,1] in R^2, A = (0.1, 0.4) a point in T, and
        B = (0.5, 0.5) the centroid of T. The diameter of T is 1.
        Then, the polynomial valued vector of A in the derivative of the scaled monomial
        basis of order 2 in T is :
        d_phi_vector = | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^0     |
                       | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^1     |
                       | 1*((0.1-0.5)/1.)^(1-1)*((0.4-0.5)/1.)^0 |
                       | 0*((0.1-0.5)/1.)^0*((0.4-0.5)/1.)^2     |
                       | 1*((0.1-0.5)/1.)^(1-1)*((0.4-0.5)/1.)^1 |
                       | 2*((0.1-0.5)/1.)^(2-1)*((0.4-0.5)/1.)^0 |
        Or equivalently :
        d_phi_vector = (1/diameter) @ gradient_operator @ phi_vector
        ================================================================================================================
        Returns :
        ================================================================================================================
        - d_phi_vector : the vector of size the dimension of the polynomial basis, whose values are the evaluation at the given point of the derivative of each monomial with respect to dx
        """
        grad_dx = self.conformal_gradients[dx]
        # print("grad_dx : {}".format(grad_dx))
        phi_vector = self.get_phi_vector(point, centroid, diameter)
        # print("phi_vector : {}".format(phi_vector))
        # print("res : {}".format(grad_dx @ phi_vector.T))
        d_phi_vector = (1.0 / diameter) * (grad_dx @ phi_vector.T)
        # print("d_phi_vector : {}".format(d_phi_vector))
        # print("---")
        return d_phi_vector
