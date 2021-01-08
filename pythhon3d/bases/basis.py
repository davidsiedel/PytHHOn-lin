from scipy.special import binom


class Basis:
    def __init__(
        self, polynomial_order: int, domain_dimension: int,
    ):
        """
        ================================================================================================================
        Class :
        ================================================================================================================
        The Basis class provides a general framework to build a polynomial basis.
        ================================================================================================================
        Parameters :
        ================================================================================================================
        - polynomial_order : the polynomial order
        - domain_dimension : the spatial dimension for the support in which polynomials act
        ================================================================================================================
        Attributes :
        ================================================================================================================
        - basis_dimension : the dimension of the polynomial basis
        """
        self.basis_dimension = int(binom(polynomial_order + domain_dimension, polynomial_order,))
