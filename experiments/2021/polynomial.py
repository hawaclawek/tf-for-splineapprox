"""Provides basic tools for polynomials.

It provides evaluation, differentiation, L2-errors, L2-best fitting polyonmials
and such. All functions are programmed in a away to play well with tensorflow.

A polynomial is encoded by the list of its coefficients with increasing degree,
i.e., [a, b, c] at x is 'a + b·x + c·x²'.
"""

__author__ = "Stefan Huber <stefan.huber@fh-salzburg.ac.at>"


import tensorflow as tf
import numpy as np


def evaluate(coeffs, x):
    """Return evaluation of a polynomial with given coefficients at location x.
    If coeffs is [a, b, c, …] then return a + b·x + c·x² + …"""

    tot = 0.0
    for c in coeffs[::-1]:
        tot = x*tot + c
    return tot

def evaluate_at_zero(coeffs):
    """Return evaluation of a polynomial with given coefficients at location 0.
    If coeffs is [a, b, c, …] then it returns a."""

    return coeffs[0]

def evaluate_vect(coeffs, xs):
    """Like a vectorized version of evaluate() for lists of values for x."""

    # We cannot use numpy.vectorize here because it would stop gradient
    # computation.
    # Note that tf.vectorize_map() is slow because len(xs) is too small
    # to pay off, I guess. Using map_fn() is similarily fast than using
    # plain list comprehension.
    return [evaluate(coeffs, x) for x in xs]

def derive(coeffs):
    """Returns the derivative of the polynomial given by the cofficients.  If
    coeffs is [a, b, c, d, …] then return [b, 2c, 3d, …]"""

    return np.arange(1, len(coeffs)) * coeffs[1:]

def l2_sq_error(coeffs, xs, ys):
    """Returns the square of the L2-error between polynomial of given coeffs
    and the samples given in ys at loctions xs. That is, if the polynomial
    given by coeffs is p then return the sum of the squares of p(x)-y where x,
    y iterates over xs, ys."""

    fs = evaluate_vect(coeffs, xs)
    ds = tf.subtract(fs, ys)
    return tf.reduce_sum(tf.multiply(ds, ds))

def l2minmizer(xs, ys, degree):
    """Returns the coefficients of the polynomial of given degree that
    minimized the l2 error for the samples given by xs and ys. That is, returns
    coeffs such that l2_sq_error(coeffx, xs, ys) is minimal."""

    return np.polyfit(xs, ys, degree)[::-1]
