"""Provides basic tools for splines.

A spline is a piecewise polynomial function. The module provides evaluation,
differentiation, L2-errors, L2-best fitting splines and such. All functions are
programmed in a away to play well with tensorflow.

This module builds upon the `polynomial` module.

We encode a spline by a pair (xlefts, polys) with a sorted list `xlefts` of
x-coordinates and a equally long list `polys` of polynomials. For an index i we
interpret xlefts[i] as the "origin" of the polynomial polys[i] in the following
sense: If the evaluation of the spline at x is given by the polynomial polys[i]
then the spline evaluation results from polynomial.evaluate(polys[i], x -
xlefts[i]).
"""

__author__ = "Stefan Huber <stefan.huber@fh-salzburg.ac.at>"


import tensorflow as tf
import numpy as np
from bisect import bisect_right

import polynomial


def find_index(xlefts, x):
    """Find polynomial index in xlefts of a spline for a given location x."""

    # bisect_right() gives insertion location in xlefts if x would be inserted, and
    # right to possible existing list entries. Hence, reducing it by one gives
    # the polynomial index.
    return max(0, bisect_right(xlefts, x) - 1)

def scan_index(xlefts, x, start=None):
    """Linear-scan vesion of find_index(xlefts, x). Starts search at given
    `start`. If `start` is None then start is None then returns find_index().
    This function is faster than find_index() if start is close to the answer.

    Requires that the answer is not less than start."""

    # There is no start so resort to quick find_index()
    if start is None:
        return find_index(xlefts, x)

    # Progress start until x is still right of xlefts[start]
    while start < len(xlefts) - 1 and x > xlefts[start + 1]:
        start += 1

    return start

def evaluate_indexed(spline, idx, x):
    """Evaluate spline polynomial of given index at location x."""

    xlefts, polys = spline
    return polynomial.evaluate(polys[idx], x - xlefts[idx])

def evaluate(spline, x):
    """Return evaluation of the spline at location x"""

    xlefts, _ = spline
    i = find_index(xlefts, x)
    return evaluate_indexed(spline, i, x)

def evaluate_vect(spline, xs):
    """Like a vectorized version of evaluate() for lists of values for x.
    Requires that xs is sorted."""

    return [evaluate(spline, x) for x in xs]

def evaluate_vect_sorted(spline, xs):
    """A faster version of evaluate_vect(), which requires that xs is sorted.
    This version is faster if len(xs) is at least in order of numer of
    polynomials."""

    res = [None] * len(xs)
    xlefts, _ = spline

    # Index of polynomial
    polyi = None

    # Run through xs and progress polyi linearily.
    for i, x in enumerate(xs):
        # Progress polyi to next x.
        polyi = scan_index(xlefts, x, polyi)
        res[i] = evaluate_indexed(spline, polyi, x)

    return res

def derive(spline):
    """Returns the derivative of the spline."""

    xlefts, polys = spline
    return xlefts, [polynomial.derive(p) for p in polys]

def l2_sq_error(spline, xs, ys):
    """Returns the square of the L2-error between spline and the samples given
    in ys at loctions xs. Requires that xs is sorted."""

    fs = evaluate_vect_sorted(spline, xs)
    ds = tf.subtract(fs, ys)
    return tf.reduce_sum(tf.multiply(ds, ds))

def partition_samples(xlefts, xs, ys):
    """Partitions the samples xs and ys according to xlefts. That is, returns a
    pair (pxs, pys) of subsequences of xs and ys such that pxs[i] and pys[i]
    belong to the i-th polynomial of a spline with given xlefts. The elements
    in pxs are numpy arrays. Requires that xs is sorted."""

    part_xs = [[] for _ in xlefts]
    part_ys = [[] for _ in xlefts]

    # Index of polynomial
    polyi = None

    for x, y in zip(xs, ys):
        # Progress polyi to next x.
        polyi = scan_index(xlefts, x, polyi)

        part_xs[polyi].append(x)
        part_ys[polyi].append(y)

    part_xs = [np.array(_) for _ in part_xs]
    return part_xs, part_ys

def l2minmizer(xs, ys, xlefts, degree):
    """Returns the spline with given xlefts that minimizes l2_sq_error(spline,
    xs, ys). Requires that xs is sorted."""

    part_xs, part_ys = partition_samples(xlefts, xs, ys)

    polys = [None] * len(xlefts)
    for i, _ in enumerate(polys):
        polys[i] = polynomial.l2minmizer(part_xs[i] - xlefts[i], part_ys[i], degree)

    return xlefts, polys

def xlefts_uniform(xs, polynum):
    """Returns xlefts for a spline with polynum-many polynomials with uniform
    supports. Requires that xs is sorted."""

    return np.linspace(xs[0], xs[-1], polynum + 1)[:-1]

def l2minmizer_uniform(xs, ys, polynum, degree):
    """Returns the spline with polynum-many polynomials with uniform supports
    that minimizes l2_sq_error(spline, xs, ys). Requires that xs is sorted."""

    xlefts = xlefts_uniform(xs, polynum)
    return l2minmizer(xs, ys, xlefts, degree)

def tf_zero_spline_uniform(xs, polynum, degree):
    """Returns the zero spline with polynum-many polynomials with uniform
    supports using tf.Variable() for the polynomial coefficients."""

    xlefts = xlefts_uniform(xs, polynum)
    polys = [[tf.Variable(0.0, dtype=tf.float64) for _ in range(degree + 1)] for _ in range(polynum)]
    return xlefts, polys

def tf_variables(spline):
    """Returns the list of tensor flow variables from a spline"""
    _, polys = spline
    return [coeff for p in polys for coeff in p]

def to_numpy(spline):
    """Returns spline with polynomials as numpy array."""
    xlefts, polys = spline
    return xlefts, np.array(polys)

def discontinuities(spline):
    """Returns a generator on the list of discontinuities of rising
    derivatives.

    When d denotes the degree of the spline (xlefts, polys), then we consider
    i-th derivative if the spline polynomial-wise for each 0 ≤ i ≤ d. (For i >
    d the derivatives are zero.) Then at each x in xlefts[1:], i.e., x =
    xlefts[k] for a 1 ≤ k < n, the respective left and right polynomial at x
    have a difference d_{ik} of i-th derivatives.

    What we return is a generator on the list
    [(d_{i1}, d_{i2}, ..., d_{i{n-1}}) for i in range(d+1)].

    Requires that xlefts is sorted and requires that all polys have same
    length."""

    xlefts, polys = spline

    # xdeltas is the length of the supports of the polynomials (except the last
    # one)
    xdeltas = [xlefts[k + 1] - xlefts[k] for k in range(len(xlefts) - 1)]

    def eval_diff(polys, k):
        """Returns difference of polynomial evaluation on xlefts[k+1] between
        polys[k] and polys[k+1]."""

        left = polynomial.evaluate(polys[k], xdeltas[k])
        right = polynomial.evaluate_at_zero(polys[k+1])
        return right - left

    degree = len(polys[0]) - 1
    i = 0
    while True:
        # Return the list for this derivative differences
        yield [eval_diff(polys, k) for k in range(len(xdeltas))]

        # Go to next derivative
        i += 1
        if i == degree + 1:
            break

        # Prepare next derivative
        polys = [polynomial.derive(p) for p in polys]
