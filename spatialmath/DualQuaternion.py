from __future__ import annotations
import numpy as np
from spatialmath import Quaternion, UnitQuaternion
from spatialmath import base
import spatialmath.pose3d as pose3d
from spatialmath.base.types import ArrayLike3, R8x8, R8

# TODO scalar multiplication


class DualQuaternion:
    r"""
    A dual number is an ordered pair :math:`\hat{a} = (a, b)` or written as
    :math:`a + \epsilon b` where :math:`\epsilon^2 = 0`.

    A dual quaternion can be considered as either:

    - a quaternion with dual numbers as coefficients
    - a dual of quaternions, written as an ordered pair of quaternions

    The latter form is used here.

    :References:

    - http://web.cs.iastate.edu/~cs577/handouts/dual-quaternion.pdf
    - https://en.wikipedia.org/wiki/Dual_quaternion

    .. warning:: Unlike the other spatial math classes, this class does not
      (yet) support multiple values per object.

    :seealso: :func:`UnitDualQuaternion`
    """

    def __init__(
        self, real: Quaternion | None = None, dual: Quaternion | None = None
    ) -> None:
        """
        Construct a new dual quaternion

        :param real: real quaternion
        :type real: Quaternion or UnitQuaternion
        :param dual: dual quaternion
        :type dual: Quaternion or UnitQuaternion
        :raises ValueError: incorrect parameters

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> print(d)
            >>> d = DualQuaternion([1, 2, 3, 4,  5, 6, 7, 8])
            >>> print(d)

        The dual number is stored internally as two quaternion, respectively
        called ``real`` and ``dual``.

        """

        if real is None and dual is None:
            self.real = None
            self.dual = None
            return
        elif dual is None and base.isvector(real, 8):
            self.real = Quaternion(real[0:4])
            self.dual = Quaternion(real[4:8])
        elif real is not None and dual is not None:
            if not isinstance(real, Quaternion):
                raise ValueError("real part must be a Quaternion subclass")
            if not isinstance(dual, Quaternion):
                raise ValueError("real part must be a Quaternion subclass")
            self.real = real  # quaternion, real part
            self.dual = dual  # quaternion, dual part
        else:
            raise ValueError("expecting zero or two parameters")

    @classmethod
    def Pure(cls, x: ArrayLike3) -> DualQuaternion:
        x = base.getvector(x, 3)
        return cls(UnitQuaternion(), Quaternion.Pure(x))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        """
        String representation of dual quaternion

        :return: compact string representation
        :rtype: str

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> str(d)
        """
        return str(self.real) + " + ε " + str(self.dual)

    def norm(self) -> tuple[float, float]:
        """
        Norm of a dual quaternion

        :return: Norm as a dual number
        :rtype: 2-tuple

        The norm of a ``UnitDualQuaternion`` is unity, represented by the dual
        number (1,0).

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.norm()  # norm is a dual number
        """
        a = self.real * self.real.conj()
        b = self.real * self.dual.conj() + self.dual * self.real.conj()
        return (base.sqrt(a.s), base.sqrt(b.s))

    def conj(self) -> DualQuaternion:
        r"""
        Conjugate of dual quaternion

        :return: Conjugate
        :rtype: DualQuaternion

        There are several conjugates defined for a dual quaternion. This one
        mirrors conjugation for a regular quaternion.  For the dual quaternion
        :math:`(p, q)` it returns :math:`(p^*, q^*)`.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.conj()
        """
        return DualQuaternion(self.real.conj(), self.dual.conj())

    def __add__(self: DualQuaternion, right: DualQuaternion) -> DualQuaternion:
        """
        Sum of two dual quaternions

        :return: Product
        :rtype: DualQuaternion

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d + d
        """
        assert self.real is not None and self.dual is not None
        assert right.real is not None and right.dual is not None

        return DualQuaternion(self.real + right.real, self.dual + right.dual)

    def __sub__(self, right: DualQuaternion) -> DualQuaternion:
        """
        Difference of two dual quaternions

        :return: Product
        :rtype: DualQuaternion

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d - d
        """
        assert self.real is not None and self.dual is not None
        assert right.real is not None and right.dual is not None

        return DualQuaternion(self.real - right.real, self.dual - right.dual)

    def __mul__(self: DualQuaternion, right: DualQuaternion) -> DualQuaternion:
        """
        Product of dual quaternion

        - ``dq1 * dq2`` is a dual quaternion representing the product of
          ``dq1`` and ``dq2``.  If both are unit dual quaternions, the product
          will be a unit dual quaternion.
        - ``dq * p`` transforms the point ``p`` (3) by the unit dual quaternion
          ``dq``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d * d
        """
        assert self.real is not None and self.dual is not None
        assert right.real is not None and right.dual is not None

        if isinstance(right, DualQuaternion):
            real = self.real * right.real
            dual = self.real * right.dual + self.dual * right.real

            if isinstance(self, UnitDualQuaternion) and isinstance(
                self, UnitDualQuaternion
            ):
                return UnitDualQuaternion(real, dual)
            else:
                return DualQuaternion(real, dual)

        elif isinstance(self, UnitDualQuaternion) and base.isvector(right, 3):
            v = base.getvector(right, 3)
            vp = self * DualQuaternion.Pure(v) * self.conj()
            return vp.dual.v

    def matrix(self) -> R8x8:
        """
        Dual quaternion as a matrix

        :return: Matrix represensation
        :rtype: ndarray(8,8)

        Dual quaternion multiplication can also be written as a matrix-vector
        product.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.matrix()
            >>> d.matrix() @ d.vec
            >>> d * d
        """
        assert self.real is not None and self.dual is not None

        return np.block(
            [[self.real.matrix, np.zeros((4, 4))], [self.dual.matrix, self.real.matrix]]
        )

    @property
    def vec(self) -> R8:
        """
        Dual quaternion as a vector

        :return: Vector represensation
        :rtype: ndarray(8)

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, Quaternion
            >>> d = DualQuaternion(Quaternion([1,2,3,4]), Quaternion([5,6,7,8]))
            >>> d.vec
        """
        assert self.real is not None and self.dual is not None

        return np.r_[self.real.vec, self.dual.vec]


class UnitDualQuaternion(DualQuaternion):
    """[summary]

    :param DualQuaternion: [description]
    :type DualQuaternion: [type]


    .. warning:: Unlike the other spatial math classes, this class does not
      (yet) support multiple values per object.

    :seealso: :func:`UnitDualQuaternion`
    """

    def __init__(
        self, real: Quaternion | None = None, dual: Quaternion | None = None
    ) -> None:
        r"""
        Create new unit dual quaternion

        :param real: real quaternion or SE(3) matrix
        :type real: Quaternion, UnitQuaternion or SE3
        :param dual: dual quaternion
        :type dual: Quaternion or UnitQuaternion

        - ``UnitDualQuaternion(real, dual)`` is a new unit dual quaternion with
          real and dual parts as specified.
        - ``UnitDualQuaternion(T)`` is a new unit dual quaternion equivalent to
          the rigid-body motion described by the SE3 value ``T``.

        Example:

        .. runblock:: pycon

            >>> from spatialmath import UnitDualQuaternion, SE3
            >>> T = SE3.Rand()
            >>> print(T)
            >>> d = UnitDualQuaternion(T)
            >>> print(d)
            >>> type(d)

        The dual number is stored internally as two quaternion, respectively
        called ``real`` and ``dual``.  For a unit dual quaternion they are
        respectively:

        .. math::

            \q_r &\sim \mat{R}

            q_d &= \frac{1}{2} q_t \q_r

        where :math:`\mat{R}` is the rotational part of the rigid-body motion
        and :math:`q_t` is a pure quaternion formed from the translational part
        :math:`t`.
        """

        if dual is None and isinstance(real, pose3d.SE3):
            T = real
            S = UnitQuaternion(T.R)
            D = Quaternion.Pure(T.t)

            real = S
            dual = 0.5 * D * S

        super().__init__(real, dual)

    def SE3(self) -> pose3d.SE3:
        """
        Convert unit dual quaternion to SE(3) matrix

        :return: SE(3) matrix
        :rtype: SE3

        Example:

        .. runblock:: pycon

            >>> from spatialmath import DualQuaternion, SE3
            >>> T = SE3.Rand()
            >>> print(T)
            >>> d = UnitDualQuaternion(T)
            >>> print(d)
            >>> print(d.T)
        """
        R = base.q2r(self.real.A)
        t = 2 * self.dual * self.real.conj()

        return pose3d.SE3(base.rt2tr(R, t.v))


if __name__ == "__main__":
    print(UnitDualQuaternion(pose3d.SE3()))
