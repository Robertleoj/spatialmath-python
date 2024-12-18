import unittest
import math
import sympy as sp

# from spatialmath.base.symbolic import *
from spatialmath.base.symbolic import (
    sin,
    cos,
    sqrt,
    simplify,
    zero,
    one,
    negative_one,
    pi,
    issymbol,
)


class Test_symbolic(unittest.TestCase):
    def test_symbol(self):
        theta = sp.symbols("theta", real=True)
        self.assertTrue(isinstance(theta, sp.Expr))
        self.assertTrue(theta.is_real)

        theta = sp.symbols("theta", real=False)
        self.assertTrue(isinstance(theta, sp.Expr))
        self.assertFalse(theta.is_real)

        theta, psi = sp.symbols("theta, psi", real=True)
        self.assertTrue(isinstance(theta, sp.Expr))
        self.assertTrue(isinstance(psi, sp.Expr))

        theta, psi = sp.symbols("theta psi", real=True)
        self.assertTrue(isinstance(theta, sp.Expr))
        self.assertTrue(isinstance(psi, sp.Expr))

        q = sp.symbols("q:6", real=True)
        self.assertEqual(len(q), 6)
        for _ in q:
            self.assertTrue(isinstance(_, sp.Expr))
            self.assertTrue(_.is_real)

    def test_issymbol(self):
        theta = sp.symbols("theta", real=True)
        self.assertFalse(issymbol(3))
        self.assertFalse(issymbol("not a symbol"))
        self.assertFalse(issymbol([1, 2]))
        self.assertTrue(issymbol(theta))

    def test_functions(self):
        theta = sp.symbols("theta", real=True)
        self.assertTrue(isinstance(sin(theta), sp.Expr))
        self.assertTrue(isinstance(sin(1.0), float))

        self.assertTrue(isinstance(cos(theta), sp.Expr))
        self.assertTrue(isinstance(cos(1.0), float))

        self.assertTrue(isinstance(sqrt(theta), sp.Expr))
        self.assertTrue(isinstance(sqrt(1.0), float))

        x = (theta - 1) * (theta + 1) - theta**2
        self.assertTrue(math.isclose(simplify(x).evalf(), -1))

    def test_constants(self):
        x = zero()
        self.assertTrue(isinstance(x, sp.Expr))
        self.assertTrue(math.isclose(x.evalf(), 0))

        x = one()
        self.assertTrue(isinstance(x, sp.Expr))
        self.assertTrue(math.isclose(x.evalf(), 1))

        x = negative_one()
        self.assertTrue(isinstance(x, sp.Expr))
        self.assertTrue(math.isclose(x.evalf(), -1))

        x = pi()
        self.assertTrue(isinstance(x, sp.Expr))
        self.assertTrue(math.isclose(x.evalf(), math.pi))


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
