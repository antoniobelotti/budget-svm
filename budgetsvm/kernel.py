"""Kernel implementation.

This module implements the kernel used in mulearn.
"""

import numpy as np


class Kernel:
    """Base kernel class."""

    def __init__(self):
        """Create an instance of :class:`Kernel`."""
        self.precomputed = False
        self.kernel_computations = None

    def compute(self, arg_1, arg_2):
        """Compute the kernel value, given two arguments.

        :param arg_1: First kernel argument.
        :type arg_1: Object
        :param arg_2: Second kernel argument.
        :type arg_2: Object
        :raises: NotImplementedError (:class:`Kernel` is abstract)
        :returns: `float` -- kernel value.
        """
        raise NotImplementedError(
            "The base class does not implement the `compute` method"
        )

    def __str__(self):
        """Return the string representation of a kernel."""
        return self.__repr__()

    def __eq__(self, other):
        """Check kernel equality w.r.t. other objects."""
        return type(self) == type(other)

    def __ne__(self, other):
        """Check kernel inequality w.r.t. other objects."""
        return not self == other

    @staticmethod
    def __nonzero__():
        """Check if a kernel is non-null."""
        return True

    def __hash__(self):
        """Generate hashcode for a kernel."""
        return hash(self.__repr__())

    @classmethod
    def get_default(cls):
        """Return the default kernel.

        :returns: `LinearKernel()` -- the default kernel.
        """
        return LinearKernel()


class LinearKernel(Kernel):
    """Linear kernel class."""

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value.

        The value $k(x_1, x_2)$ of a linear kernel is equal to the dot product
        $x_1 \cdot x_2$, that is to $\sum_{i=1}^n (x_1)_i (x_2)_i$, $n$ being
        the common dimension of $x_1$ and $x_2$.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        return float(np.dot(arg_1, arg_2))

    def __repr__(self):
        """Return the python representation of the kernel."""
        return "LinearKernel()"


class PolynomialKernel(Kernel):
    """Polynomial kernel class."""

    def __init__(self, degree):
        r"""Create an instance of `PolynomialKernel`.

        :param degree: degree of the polynomial kernel.
        :type degree: `int`
        :raises: ValueError if `degree` is not an integer or if it has a
          negative value.
        """
        super().__init__()
        if degree > 0 and isinstance(degree, int):
            self.degree = degree
        else:
            raise ValueError(f"{degree} is not usable as a polynomial degree")

    def compute(self, arg_1, arg_2):
        r"""
        Compute the kernel value.

        The value $k(x_1, x_2)$ of a polynomial kernel is equal to the
        quantity $(x_1 \cdot x_2 + 1)^d$, $d$ being the polynomial degree of
        the kernel.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        return float((np.dot(arg_1, arg_2) + 1) ** self.degree)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"PolynomialKernel({self.degree})"

    def __eq__(self, other):
        return isinstance(other, PolynomialKernel) and self.degree == other.degree


class HomogeneousPolynomialKernel(PolynomialKernel):
    """Homogeneous polynomial kernel class."""

    def __init__(self, degree):
        r"""Create an instance of `HomogeneousPolynomialKernel`.

        :param degree: degree of the polynomial kernel.
        :type degree: `int`
        :raises: ValueError if `degree` is not an integer or if it has a
          negative value.
        """
        super().__init__(degree)

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a homogeneous polynomial kernel is
        intended as the quantity $(x_1 \cdot x_2)^d$, $d$ being the polynomial
        degree of the kernel.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        return float(np.dot(arg_1, arg_2) ** self.degree)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"HomogeneousPolynomialKernel({self.degree})"

    def __eq__(self, other):
        return isinstance(other, PolynomialKernel) and self.degree == other.degree


class GaussianKernel(Kernel):
    """Gaussian kernel class."""

    default_sigma = 1

    def __init__(self, sigma=default_sigma):
        r"""Create an instance of `GaussianKernel`.

        :param sigma: gaussian standard deviation, defaults to 1.
        :type sigma: `float`
        :raises: ValueError if `sigma` has a negative value.
        """
        super().__init__()
        if sigma > 0:
            self.sigma = sigma
        else:
            raise ValueError(
                f"{sigma} is not usable " "as a gaussian standard deviation"
            )

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a gaussian kernel is intended as the
        quantity $\mathrm e^{-\frac{||x_1 - x_2||^2}{2 \sigma^2}}$, $\sigma$
        being the kernel standard deviation.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        diff = np.linalg.norm(np.array(arg_1) - np.array(arg_2)) ** 2
        return float(np.exp(-1.0 * diff / (2 * self.sigma**2)))

    def __repr__(self):
        """Return the python representation of the kernel."""
        obj_repr = "GaussianKernel("
        if self.sigma != self.default_sigma:
            obj_repr += f"sigma={self.sigma}"
        obj_repr += ")"
        return obj_repr

    def __eq__(self, other):
        return isinstance(other, GaussianKernel) and self.sigma == other.sigma


class HyperbolicKernel(Kernel):
    """Hyperbolic kernel class."""

    default_scale = 1
    default_offset = 0

    def __init__(self, scale=default_scale, offset=default_offset):
        r"""Create an instance of `HyperbolicKernel`.

        :param scale: scale constant, defaults to 1.
        :type scale: `float`
        :param offset: offset constant, defaults to 0.
        :type offset: `float`
        """
        super().__init__()
        self.scale = scale
        self.offset = offset

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value $k(x_1, x_2)$ of a hyperbolic kernel is intended as the
        quantity $\tanh(\alpha x_1 \cdot x_2 + \beta)$, $\alpha$ and $\beta$
        being the scale and offset parameters, respectively.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        dot_orig = np.dot(np.array(arg_1), np.array(arg_2))
        return float(np.tanh(self.scale * dot_orig + self.offset))

    def __repr__(self):
        """Return the python representation of the kernel."""
        obj_repr = "HyperbolicKernel("
        if self.scale != self.default_scale:
            obj_repr += f"scale={self.scale}, "
        if self.offset != self.default_offset:
            obj_repr += f"offset={self.offset}, "

        if obj_repr.endswith(", "):
            return obj_repr[:-2] + ")"
        else:
            return "HyperbolicKernel()"

    def __eq__(self, other):
        return (
            isinstance(other, HyperbolicKernel)
            and self.scale == other.scale
            and self.offset == other.offset
        )


class PrecomputedKernel(Kernel):
    """Precomputed kernel class."""

    def __init__(self, original_kernel = GaussianKernel()):
        r"""Create an instance of `PrecomputedKernel`.
        """
        super().__init__()
        self.precomputed = True
        self.original_kernel = original_kernel

    def compute(self, arg_1, arg_2):
        r"""Compute the kernel value.

        The value of a precomputed kernel is retrieved according to the indices
        of the corresponding objects. Note that each index should be enclosed
        within an iterable in order to be compatible with sklearn.

        :param arg_1: First kernel argument.
        :type arg_1: iterable of `float`
        :param arg_2: Second kernel argument.
        :type arg_2: iterable of `float`
        :returns: `float` -- kernel value.
        """
        return self.original_kernel.compute(arg_1, arg_2)

    def __repr__(self):
        """Return the python representation of the kernel."""
        return f"PrecomputedKernel({self.original_kernel.__repr__()})"