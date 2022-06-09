# !/usr/bin/env python

"""model.py: Fitting polynomial splines of arbitrary degree and C^k-continuity.
             Perform optimization using TensorFlow GradientTape environment.  
"""
import numpy
import numpy as np
from numpy.core.function_base import linspace
import sklearn.preprocessing as preprocessing
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import random
import math
import statistics
import time
import copy
from tensorflow.python.keras.backend import _fused_normalize_batch_in_training

__author__ = "Hannes Waclawek"
__version__ = "2.1"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"

# Maximum values
POLY_DEGREE_MAX = 19
POLY_NUM_MAX = 128

# Default constructor values
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 500
DEFAULT_OVERLAP = 0
DEFAULT_CURVATURE_FACTOR = 0
DEFAULT_CK_PRESSURE_FACTOR = 0
DEFAULT_APPROXIMATION_QUALITY_FACTOR = 1

# Other Constants
SHIFT_POLYNOMIAL_CENTER_MEAN = 0
SHIFT_POLYNOMIAL_CENTER_BOUNDARY = 1
SHIFT_POLYNOMIAL_CENTER_OFF = 2


def get_spline_from_coeffs(coeffs, data_x, data_y, uniform_split = False, shift_polynomial_centers='mean',
                           total_loss_values = None, e_loss_values = None, D_loss_values = None):
    '''Generate Spline from existing spline coefficients'''
    s = Spline(polydegree=coeffs[0].get_shape()[0] - 1, polynum=len(coeffs))
    s.coeffs = coeffs
    s.data_x = data_x
    s.data_y = data_y
    s.performedfit = True

    if shift_polynomial_centers is None:
        s.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_OFF
    elif shift_polynomial_centers == 'mean':
        s.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
    elif shift_polynomial_centers == 'boundary':
        s.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_BOUNDARY
    else:
        s.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_OFF

    if uniform_split:
        s._get_boundary_points_uniform()
    else:
        s._get_boundary_points_non_uniform()
    s._split_data()

    if total_loss_values is not None:
        s.total_loss_values = total_loss_values
        s.epochs = len(total_loss_values)
    s.e_loss_values = e_loss_values
    s.D_loss_values = D_loss_values

    return s


class Spline():
    """Main class containing coefficients and methods to fit and evaluate splines"""

    def __init__(self, polydegree, polynum, ck=2, clamped=True, periodic=False):
        """
        :param polydegree: Polynomial order of individual functional pieces
        :param polynum: Number of polynomial pieces
        :param ck: C^k-continuity, e.g. 2 = C^2-continuous spline. Requires minimum polydegree 2k + 1
        :param periodic: "True" --> Align derivatives at position xn with derivatives at position x0. If clamped = true, derivative 0 will not be aligned
        :param clamped: "True" --> Clamp spline result (derivative 0) to first and last data point of input space
        """
        # Spline parameters
        self.polydegree = polydegree  # Polynomial order of individual functional pieces
        self.polynum = polynum  # Number of polynomial pieces
        self.ck = ck
        self.clamped = clamped
        self.periodic = periodic

        if polynum <= 0 or polynum > POLY_NUM_MAX:
            raise Exception("Invalid polynomial count - Must be 1 <= polynum <= " + str(POLY_NUM_MAX))

        if polydegree <= 0 or polydegree > POLY_DEGREE_MAX:
            raise Exception("Invalid polynomial degree - Must be 1 <= polydegree <= " + str(POLY_DEGREE_MAX))

        if polydegree < 2 * ck + 1:
            raise Exception("C^" + str(ck) + "-continuous spline requires minimum polydegree " + str(2 * ck + 1))

        # Parameters
        self.epochs = DEFAULT_EPOCHS
        self.overlap_segments = DEFAULT_OVERLAP
        self.gradient_regularization = False
        self.factor_ck_pressure = DEFAULT_CK_PRESSURE_FACTOR
        self.factor_approximation_quality = DEFAULT_APPROXIMATION_QUALITY_FACTOR
        self.factor_curvature = DEFAULT_CURVATURE_FACTOR

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)

        self.initialize_l2fit = False
        self.continuity = True
        self.verbose = False
        self.shift_polynomial_centers = 0

        # Data arrays
        self.data_x = []
        self.data_y = []
        self.data_x_split = [[]]
        self.data_y_split = [[]]
        self.data_x_split_overlap = [[]]
        self.data_y_split_overlap = [[]]
        self.boundary_points = []

        # Internal variables
        self.performedfit = False
        self.initial_data_array_length = 0

        # Ck-pressure scaling
        self.pk = []  # Array holding peak values of C^k relevant derivatives

        # TF oefficients array
        self.coeffs = [None for _ in range(self.polynum)]

        # Initialize loss arrays (For plotting loss functions after optimization)
        self.total_loss_values = []  # combined loss
        self.I_loss_values = []  # loss for integrate_squared_spline_acceleration()
        self.D_loss_values = []  # loss for ck_pressure() - total
        self.d_loss_values = [[]]  # loss for ck_pressure() - loss_d
        self.e_loss_values = []  # loss for sum of squared approximation errors

    def properties(self):
        print(f'Polynomial degree: {self.polydegree}')
        print(f'Number of polynomial segments: {self.polynum}')
        print(f'Clamped: {self.clamped}')
        print(f'Segment overlap: {self.overlap_segments}')
        print(f'C{self.ck}-continuous')
        print(f'Optimization learning rate: {self.learning_rate}')
        print(f'Optimization epochs: {self.epochs}')

    def _sum_squared_errors(self):
        """Calculate sum of squared errors (derivative 0) with regards to self.data_y.
        Invariant to length of data points array.       
        :return: derivative 0 squared error"""
        total_cost = 0

        for i in range(self.polynum):
            y1 = self._evaluate_polynomial_at_x(i, 0, self.data_x_split[i], self._polynomial_center(i))
            #y1 = self._evaluate_polynomial_at_x(i, 0, self.data_x_split_overlap[i], self._polynomial_center(i))
            y2 = self.data_y_split[i]
            #y2 = self.data_y_split_overlap[i]
            e = tf.subtract(y1, y2)
            total_cost += tf.reduce_sum(tf.multiply(e,e))

        return tf.multiply(tf.divide(total_cost, len(self.data_x)), self.polynum)
        #return tf.divide(total_cost, len(self.data_x)) * self.polynum

    def ck_pressure(self):
        """Cost function: "C^k-pressure"
        (= distance between segment endpoints in continuous derivatives (see self.ck))       
        :return: Sum of total C^k error values (Invariant to number of boundary points), C^k error array for self.ck derivatives"""
        if not self.performedfit:
            raise Exception("No spline data - Perform fit() first")

        total_cost = 0
        cost_d = [0.0] * (self.ck + 1)

        for k in range(self.ck + 1):
            for i in range(self.polynum - 1):
                y1 = self._evaluate_polynomial_at_x(i, k, self.boundary_points[i+1],
                                                    self._polynomial_center(i))
                y2 = self._evaluate_polynomial_at_x(i + 1, k, self.boundary_points[i+1],
                                                    self._polynomial_center(i + 1))

                # Experiments show, that this equilibration is not needed
                # if self.pk[k] > 0:
                #     total_cost += tf.square((y1 - y2)/self.pk[k])
                #     cost_d[k] += tf.square((y1 - y2)/self.pk[k])
                # else:
                total_cost += tf.square((y1 - y2))
                cost_d[k] += tf.square((y1 - y2))

        return tf.divide(total_cost, (len(self.boundary_points)-2)), cost_d # divide with number of boundary points (= boundary points except first and last one)

    def integrate_squared_spline_acceleration(self):
        """ Perform integration of stored spline coefficients' 2nd derivative
            :return: total curvature value of spline
        """
        if not self.performedfit:
            raise Exception("No spline data - Perform fit() first")

        ret = 0

        for i in range(0, self.polynum):
            x0 = self._polynomial_center(i)
            for j in range(2, self.polydegree + 1):
                for k in range(2, self.polydegree + 1):
                    ret += self.coeffs[i][j] * self.coeffs[i][k] * ((j * k * (j - 1) * (k - 1)) / (j + k - 3)) * (
                                (self.data_x_split[i][-2] - x0) ** (j + k - 3) - (self.data_x_split[i][0] - x0) ** (
                                    j + k - 3))

        return ret

    def _get_boundary_points_uniform(self):
        """Equally divides input space"""
        self.boundary_points = np.linspace(self.data_x[0], self.data_x[-1], self.polynum + 1)

    def _get_boundary_points_non_uniform(self):
        """Boundary points lie on input points"""
        if self.polynum <= 0:
            raise Exception("Invalid polynomial count")

        if self.polynum == 1:
            self.boundary_points = [self.data_x[0], self.data_x[-1]]
            return

        data_x_split = np.array_split(self.data_x, self.polynum)

        self.boundary_points.append(data_x_split[0][0])

        for i in range(len(data_x_split)):
            self.boundary_points.append(data_x_split[i][-1])

    def _split_input_arrays_with_boundary_points(self, input_arr):
        output_arr = [[]]
        i = 1
        j = 0
        while i < len(self.boundary_points):
            while (self.data_x[j] < self.boundary_points[i]) or (math.isclose(self.data_x[j], self.boundary_points[i])):
                output_arr[i-1].append(input_arr[j])
                j += 1
                if j > (len(input_arr) - 1):
                    break
            i += 1
            if i < len(self.boundary_points):
                output_arr.append([])

        # create list (since length of arrays may vary and numpy array of arrays requires uniform length) of numpy arrays
        # we require numpy arrays in evaluate_polynomial_at_x()
        ret = []
        for segment in output_arr:
            ret.append(np.array(segment, dtype=segment[0]))

        return ret

    def _split_data(self, overlap=0):
        """Split data into POLY_NUM segments. Only used for overlapping segments.
        :param overlap: Specify number of points of segment n to be contained in adjacent segments n-1 and n+1
                        in order to better align the curve's derivatives at transition points.
                        (Reduce "C^k-pressure")
                        Must be in the range 0 < overlap <= 1.

                        Examples:
                        overlap = 1: All points of segment n are contained in adjacent segments n-1 and n+1
                        overlap = 0.5: 50% of points of segment n are contained in adjacent segments n-1 and n+1
                        overlap = 0.1: 10% of points of segment n are contained in adjacent segments n-1 and n+1
                        Result is rounded up
        """
        if overlap < 0 or overlap > 1:
            raise Exception("Overlap must be between 0 and 1")

        if self.polynum <= 0:
            raise Exception("Invalid polynomial count")

        if self.polynum == 1:
            self.data_x_split[0] = self.data_x
            self.data_y_split[0] = self.data_y
            self.data_x_split_overlap = self.data_x_split.copy()
            self.data_y_split_overlap = self.data_y_split.copy()
            return

        # 2 Data points per segment required
        if (len(self.data_x) / self.polynum) < 2:
            raise Exception("Not enough data points for polynomial count")

        # boundary points initialization required
        if len(self.boundary_points) < 2:
            raise Exception("Internal error: Boundary points not defined")

        # split into chunks
        self.data_x_split = self._split_input_arrays_with_boundary_points(self.data_x)
        self.data_y_split = self._split_input_arrays_with_boundary_points(self.data_y)

        # # split into chunks
        # self.data_x_split = np.array_split(self.data_x, self.polynum)
        # self.data_y_split = np.array_split(self.data_y, self.polynum)
        #
        # self.initial_data_array_length = len(self.data_x_split[0])

        overlapping_points = int(math.ceil(len(self.data_x_split[0]) * overlap))

        if overlapping_points > 0:
            # add segment 1 with overlapping points to the right
            self.data_x_split_overlap = [np.append(self.data_x_split[0],
                                                             self.data_x_split[1][0: overlapping_points], axis=0)]
            self.data_y_split_overlap = [np.append(self.data_y_split[0],
                                                             self.data_y_split[1][0: overlapping_points], axis=0)]

            for i in range(1, len(self.data_x_split)):
                # add last segment with overlapping points to the left
                if i == (len(self.data_x_split) - 1):
                    self.data_x_split_overlap.append(np.insert(self.data_x_split[i], 0,
                                                             self.data_x_split[i - 1][-1 * overlapping_points:], axis=0))
                    self.data_y_split_overlap.append(np.insert(self.data_y_split[i], 0,
                                                             self.data_y_split[i - 1][-1 * overlapping_points:], axis=0))
                # add segments with overlapping points on both sides
                else:
                    add_x = np.append(self.data_x_split[i], self.data_x_split[i + 1][0: overlapping_points], axis=0)
                    add_x = np.insert(add_x, 0, self.data_x_split[i - 1][-1 * overlapping_points:], axis=0)
                    self.data_x_split_overlap.append(add_x)

                    add_y = np.append(self.data_y_split[i], self.data_y_split[i + 1][0: overlapping_points], axis=0)
                    add_y = np.insert(add_y, 0, self.data_y_split[i - 1][-1 * overlapping_points:], axis=0)
                    self.data_y_split_overlap.append(add_y)

        else:
            # create overlapping arrays
            self.data_x_split_overlap = self.data_x_split.copy()
            self.data_y_split_overlap = self.data_y_split.copy()

    def _linear_equationsystem_x_values(self, x, x0, degree, deriv):
        """Retrieve left hand side vector for polynomial of given degree
        :param x: Single x- value for which the return value should be calculated
        :param x0: Mean of given x-interval = center of polynomial piece
        :param degree: degree of the polynomial
        :param deriv: derivative of the polynomial
        :return: As an example, degree 3, derivative 0 will return [1, x, x**2, x**3]
        """
        if deriv < 0:
            raise Exception("Negative derivative")

        if deriv == 0:
            return [(x - x0) ** i for i in range(degree + 1)]

        result = []

        for i in range(degree + 1):
            if (i - deriv >= 0):
                result.append(math.factorial(i) / math.factorial(i - deriv) * (x - x0) ** (i - deriv))
            else:
                result.append(0)

        return result

    def _evaluate_polynomial_at_x(self, polynum, deriv, x, x0):
        """Return evaluation of a polynomial with given coefficients at location x using Horner's method.
        :param polynum: number of polynomial piece to evaluate. e.g. 0 will use self.coeffs[0] 
        :param deriv: derivative of the polynomial (deriv 0 = y, deriv 1 = y', ...))
        :param x: Single x value or x-value ndarray. Has to be within limits of polynomial segment
        :param x0: center of polynomial piece
        :return: value of a0 + (x-x0)*(a1 + (x-x0)*(a2+(x-x0)*(a3+...+(x-x0)(a_{n-1}+x*a_n))))
        """
        # # bad way of doing it, since pow operation is involved
        # However, Tensorflow cannot compute gradients for CK-pressure loss for derivatives > 0
        # using _polynomial_derive(), so we stick to this evaluation method
        # result = []

        # for i in range(self.polydegree + 1):
        #     if(i - deriv >= 0):
        #         result.append(math.factorial(i)/math.factorial(i - deriv) * self.coeffs[polynum][i] * (x-x0)**(i - deriv))

        # return sum(result)
        coeffs = self.coeffs[polynum]

        if deriv == 0:
            pass
        elif deriv < 0 or deriv >= self.polydegree:
            raise Exception("Invalid derivative parameter")
        else:
            for i in range(deriv):
                coeffs = self._polynomial_derive(coeffs)
        if type(x) != numpy.ndarray and type(x) != numpy.float64 and type(x) != numpy.float32:
            raise Exception(f'Expected input to be of numpy.array or numpy.float but is {type(x)}')

        res = 0.0

        for c in coeffs[::-1]:
            res = (x - x0) * res + c

        return res

    def _polynomial_derive(self, coeffs):
        """Returns the derivative of the polynomial given by the cofficients.
        :param coeffs: polynomial coefficients
        :return: If coeffs is [a, b, c, d, …] then return [b, 2c, 3d, …]"""
        return np.arange(1, coeffs.get_shape()[0]) * coeffs[1:]

    def evaluate_spline_at_x(self, x, deriv=0):
        """Evaluate spline at position x
        :param x: Single x value or x-value array. Has to be within limits of self.data_x
        :param deriv: derivative of the spline (deriv 0 = y, deriv 1 = y', ...))
        :return: value of a0 + (x-x0)*(a1 + (x-x0)*(a2+(x-x0)*(a3+...+(x-x0)(a_{n-1}+x*a_n))))
        """
        if deriv < 0:
            raise Exception("Negative derivative")

        if len(self.coeffs) == 0:
            raise Exception("No spline data - Perform fit() first")

        # Evaluate singe x value
        if np.size(x) == 1:
            segment = 0

            if x < self.data_x[0] or x > self.data_x[-1]:
                raise Exception("Provided x-value not in range [" + str(self.data_x[0]) + "," + str(
                    self.data_x[-1]) + "]")

            for i in range(self.polynum):
                if x < self.data_x_split[i][0]:
                    break
                segment = i

            y = self._evaluate_polynomial_at_x(segment, deriv, x, self._polynomial_center(segment))

        # Evaluate x - array
        elif np.size(x) > 1:
            if not self._strictly_increasing(x):
                raise Exception("x - vector not strictly increasing")

            if x[0] < self.data_x_split[0][0] or x[-1] > self.data_x_split[self.polynum - 1][-1]:
                raise Exception(f'Provided x-values in range [{x[0]}, {x[-1]}], Spline expected range [{self.data_x_split[0][0]}, {self.data_x_split[self.polynum - 1][-1]}]')

            begin_segment = 0
            end_segment = 0

            x_split = [[]] * self.polynum
            i = 0
            j = 0

            # Assign given x-values to spline segments
            while i < self.polynum:
                x_split[j] = []
                endpoint = self.boundary_points[i+1]
                while x[end_segment] <= endpoint:
                    end_segment += 1
                    if end_segment >= len(x):
                        break

                if begin_segment != end_segment:
                    x_split[j] = x[begin_segment:end_segment]
                    begin_segment = end_segment

                if end_segment >= len(x) - 1:
                    # if j == self.polynum - 2:  # if last segment only contains one element
                    #     x_split[j + 1] = [x[-1]]
                    break

                j += 1
                i += 1

            y = []

            # Evaluate x segments and merge to result vector
            for segment in range(self.polynum):
                if len(x_split[segment]) != 0:
                    data = self._evaluate_polynomial_at_x(segment, deriv, x_split[segment],
                                                          self._polynomial_center(segment))
                    y.extend(data)
        else:
            raise Exception("No x-value(s) provided")

        return y

    def _polynomial_center(self, polynomial_index):
        if self.shift_polynomial_centers is None:
            return 0
        elif self.shift_polynomial_centers == SHIFT_POLYNOMIAL_CENTER_MEAN: # mean
            return statistics.mean([self.boundary_points[polynomial_index], self.boundary_points[polynomial_index + 1]])
        elif self.shift_polynomial_centers == SHIFT_POLYNOMIAL_CENTER_BOUNDARY: # left boundary point
            return self.boundary_points[polynomial_index]
        else:
            return 0

    def _strictly_increasing(self, L):
        """Check if elements in L are strictly increasing
        :param: L: vector of elements
        :return: true if elements in L are strictly increasing
        """
        return all(x < y for x, y in zip(L, L[1:]))

    def _establish_continuity(self):
        """Align polynomial ends with "self.ck"-continuity.
        C^k-continuity, e.g. 2 = C^2-continuous spline
        Requires minimum polydegree 2k + 1 (needed_coeffs = 2(k+1), needed_degree=needed_coeffs-1)
        """

        if self.polynum < 2:
            return

        i = 0

        no_equations_per_boundary_point = self.ck + 1
        corr_poly_degree = 2 * self.ck + 1

        # Build from left to right
        while i < self.polynum:

            # Endpoints for polynomial
            x1 = self.data_x_split[i][0]
            y_1 = self.data_y_split[i][0]
            x2 = self.data_x_split[i][-1]
            y_2 = self.data_y_split[i][-1]
            x0 = self._polynomial_center(i)

            # Arrays holding derivatives at point x1 and x2 = equation system right hand side
            y1 = []
            y2 = []

            # Equation system left hand side
            a = []

            # Construct corrective polynomial points
            # Derivatives and equation system for left boundary point
            for k in range(no_equations_per_boundary_point):
                if i == 0:
                    if self.periodic:
                        if self.clamped and k == 0:  # Overtake y-value of derivative 0 if clamped
                            target = self.data_y_split[0][0]
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            target = self._evaluate_polynomial_at_x(self.polynum - 1, k,
                                                                    self.data_x_split[self.polynum - 1][-1],
                                                                    self._polynomial_center(self.polynum - 1)).numpy()  # Overtake boundary derivative value of last segment
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                    elif self.clamped:
                        if (k == 0):
                            target = self.data_y_split[0][0]  # Overtake first y-value
                            diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            diff = 0  # Leave all other derivative values
                    else:
                        diff = 0  # Leave current derivative values
                else:
                    target = self._evaluate_polynomial_at_x(i - 1, k, x1, self._polynomial_center(i - 1)).numpy()  # Overtake boundary derivative value of previous segment
                    diff = self._evaluate_polynomial_at_x(i, k, x1, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point

                y1.append(diff)
                a.append(self._linear_equationsystem_x_values(x1, x0, corr_poly_degree, k))

            # Derivatives and equation system for right boundary point
            for k in range(no_equations_per_boundary_point):
                if i == self.polynum - 1:
                    if self.periodic:
                        if self.clamped and k == 0:  # Overtake y-value of derivative 0 if clamped
                            target = self.data_y_split[-1][-1]
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            target = self._evaluate_polynomial_at_x(0, k, self.data_x_split[0][0], self._polynomial_center(0)).numpy()  # Overtake boundary derivative value of first segment
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                    elif self.clamped:
                        if k == 0:
                            target = self.data_y_split[-1][-1]  # Overtake last y-value
                            diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) - target  # Calculate difference to current value of polynomial at boundary point
                        else:
                            diff = 0  # Leave all other derivative values
                    else:
                        diff = 0  # Leave current derivative values
                else:
                    target = ((self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i))
                               + self._evaluate_polynomial_at_x(i + 1, k, x2, self._polynomial_center(i + 1))) / 2).numpy()  # Calculate mean of boundary derivative values
                    diff = self._evaluate_polynomial_at_x(i, k, x2, self._polynomial_center(i)) \
                           - target  # Calculate difference to current value of polynomial at boundary point

                y2.append(diff)
                a.append(self._linear_equationsystem_x_values(x2, x0, corr_poly_degree, k))

            # Solve resulting equation system
            b = y1 + y2
            t = np.linalg.solve(a, b)

            p1 = poly.Polynomial(t)  # corrective polynomial
            p2 = poly.Polynomial(self.coeffs[i].numpy())  # current polynomial
            p3 = poly.polysub(p2, p1)[0]  # difference
            c = p3.coef

            if len(c) < (self.polydegree + 1):
                c = np.pad(c, (0, self.polydegree + 1 - len(c)), constant_values=(0))

            self.coeffs[i] = tf.Variable(c)  # Update coefficients

            i += 1

    def _trans_probabilities(self, xs):
        """Takes a sequence of numbers and makes it sum up to one.
        :param: xs: Input x-vector.
        :return: normalized x-vector"""
        xs = np.array(xs)
        return xs / sum(xs)

    def _initialize_spline_data(self):
        """Split input data into POLY_NUM segments and initialize coefficients"""
        self._split_data(self.overlap_segments)
        self._initialize_polynomial_coefficients()

    def _initialize_polynomial_coefficients(self):
        """Initialize polynomial coefficients as tf.Variables
        If self.initialize_l2fit = True, coefficients will be initialized with least sqaure fit, otherwise with 0."""
        for i in range(self.polynum):
            if self.initialize_l2fit:
                x0 = self._polynomial_center(i)
                coeff = poly.polyfit(self.data_x_split_overlap[i] - x0, self.data_y_split_overlap[i], self.polydegree)
                self.coeffs[i] = tf.Variable(coeff, dtype='float64', trainable=True)
            else:
                self.coeffs[i] = tf.Variable([0.0 for _ in range(self.polydegree + 1)], dtype='float64', trainable=True)

    def _optimize_spline(self):
        """Optimize cost function
        self.factor_curvature * self.integrate_squared_spline_acceleration() + self.factor_ck_pressure * self.ck_pressure() 
        +  self.factor_approximation_quality * self._sum_squared_errors()
        Factors have to sum up to 1, otherwise the configured learning rate is increased effectively."""
        total_cost_value = 0.0
        cost_I = 0.0
        cost_D = 0.0
        cost_d = []
        reg_grads = [0] * (self.polynum)
        gradients = [0] * (self.polynum)
        self.total_loss_values = [0.0] * (self.epochs)
        self.D_loss_values = [0.0] * (self.epochs)
        self.d_loss_values = [0.0] * (self.epochs)
        self.I_loss_values = [0.0] * (self.epochs)
        self.e_loss_values = [0.0] * (self.epochs)

        if self.verbose:
            print("TensorFlow: Number of recognized GPUs: ", len(tf.config.list_physical_devices('GPU')))

        # Gradient regularization depending on degree of coefficient
        for j in range(self.polynum):
            if self.gradient_regularization:
                reg_grads[j] = [1.0 / (1 + i) for i in range(self.polydegree + 1)]
            else:
                reg_grads[j] = np.ones(self.polydegree + 1)
            # Make gradient regularization coefficients a probability distribution.
            # This makes the sum of gradients independent of degree.
            reg_grads[j] = self._trans_probabilities(reg_grads[j])

        # Fitting optimization routine
        for epoch in range(self.epochs):
            # Gradient Tape
            with tf.GradientTape(persistent=True) as tape:
                cost_I = tf.constant(0.0, dtype='float64')
                cost_D = tf.constant(0.0, dtype='float64')
                cost_e = tf.constant(0.0, dtype='float64')
                if self.factor_curvature > 0:
                    cost_I = tf.multiply(self.integrate_squared_spline_acceleration(), self.factor_curvature)
                if self.factor_ck_pressure > 0:
                    cost_D, cost_d = self.ck_pressure()
                if cost_D != 0:
                    cost_D = tf.multiply(cost_D, self.factor_ck_pressure)
                if self.factor_approximation_quality > 0:
                    cost_e = self._sum_squared_errors()
                if cost_e != 0:
                    cost_e = tf.multiply(cost_e, self.factor_approximation_quality)
                total_cost_value = tf.add(cost_I, tf.add(cost_D, cost_e))

            cost_d = [item * self.factor_ck_pressure for item in cost_d]

            # Save cost_values for loss plot
            self.total_loss_values[epoch] = total_cost_value
            self.D_loss_values[epoch] = cost_D
            self.d_loss_values[epoch] = cost_d
            self.I_loss_values[epoch] = cost_I
            self.e_loss_values[epoch] = cost_e

            # Calculate Gradients
            gradients = tape.gradient(total_cost_value, self.coeffs)

            # Apply regularization
            for i in range(self.polynum):
                gradients[i] = gradients[i] * reg_grads[i]

            # Apply Gradients
            self.optimizer.apply_gradients(zip(gradients, self.coeffs))

            if self.verbose and epoch % 10 == 0:
                # print("Gradients epoch ", epoch, ": ", gradients, "\n")
                print("epoch=%d, loss=%4g\r" % (epoch, total_cost_value), end="")

    def fit(self, data_x, data_y, overlap_segments=0, initialize_l2fit=False,
            uniform_split=False, shift_polynomial_centers='mean', **kwargs):
        """Fits spline to data_x / data_y using specified Hyperparameters and returns cost value
        :param: Data x-values. Have to be increasing.
        :param: Data y-values. 
        :param: overlap_segments: Only relevant if initialize_l2fit == True.
                                  Percentage of adjacent points of segment n to be included in
                                  adjacent segments n-1 and n+1 for initial l2 fit
                                  in order to better align the curve's derivatives at transition points.
        :param: initialize_l2fit: Perform least squares fitting initialization.
        :param: shift_polynomial_centers: Shift center of polynomial evaluation to the right.
                'mean': Shift center of polynomial evaluation to mean of segment
                'boundary': Shift center of polynomial evaluation to left boundary point of segment
                'off': Do not perform re-centering
        **kwargs: Parameters for optimization. See method optimization() for details
        :return: Loss value at the end of optimization
        """
        self.data_x = data_x
        self.data_y = data_y
        self.overlap_segments = overlap_segments
        self.initialize_l2fit = initialize_l2fit

        if uniform_split:
            self._get_boundary_points_uniform()
        else:
            self._get_boundary_points_non_uniform()

        if shift_polynomial_centers is None:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_OFF
        elif shift_polynomial_centers == 'mean':
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_MEAN
        elif shift_polynomial_centers == 'boundary':
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_BOUNDARY
        else:
            self.shift_polynomial_centers = SHIFT_POLYNOMIAL_CENTER_OFF

        if overlap_segments < 0 or overlap_segments > 1:
            raise Exception("Invalid overlap_segments parameter - Must be 0 <= overlap_segments <= 1")

        self._initialize_spline_data()

        self.performedfit = True

        if self.continuity:
            self._establish_continuity()

        # Perform optimization
        if 'optimizer' in kwargs:
            optimizer = kwargs['optimizer']
        else:
            optimizer = self.optimizer
        if 'n_epochs' in kwargs:
            n_epochs = kwargs['n_epochs']
        else:
            n_epochs = DEFAULT_EPOCHS
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        else:
            learning_rate = DEFAULT_LEARNING_RATE
        if 'factor_ck_pressure' in kwargs:
            factor_ck_pressure = kwargs['factor_ck_pressure']
        else:
            factor_ck_pressure = DEFAULT_CK_PRESSURE_FACTOR
        if 'factor_approximation_quality' in kwargs:
            factor_approximation_quality = kwargs['factor_approximation_quality']
        else:
            factor_approximation_quality = DEFAULT_APPROXIMATION_QUALITY_FACTOR
        if 'factor_curvature' in kwargs:
            factor_curvature = kwargs['factor_curvature']
        else:
            factor_curvature = DEFAULT_CURVATURE_FACTOR
        if 'gradient_regularization' in kwargs:
            gradient_regularization = kwargs['gradient_regularization']
        else:
            gradient_regularization = False

        if n_epochs > 0:
            self.optimize(optimizer=optimizer, n_epochs=n_epochs, learning_rate=learning_rate,
                          factor_ck_pressure=factor_ck_pressure,
                          factor_approximation_quality=factor_approximation_quality, factor_curvature=factor_curvature,
                          gradient_regularization=gradient_regularization)

            return self.total_loss_values[-1]
        else:
            return None

    def optimize(self, optimizer, n_epochs=DEFAULT_EPOCHS, **kwargs):
        """Optimizes spline from previous fit(data_x, data_y) using TensorFlow Gradient Tapes
        Cost function:
        self.factor_curvature * self.integrate_squared_spline_acceleration() + self.factor_ck_pressure * self.ck_pressure() 
        +  self.factor_approximation_quality * self._sum_squared_errors()
        Factors have to sum up to 1, otherwise the configured learning rate is increased effectively.
        :param: optimizer: Optimizer to use (keras.optimizers object)
        :param: n_epochs: Number of optimization cycles.
        :param: factor_ck_pressure: Equilibration factor for Ck-pressure
        :param: factor_approximation_quality: Equilibration factor for approximation quality
        :param: factor_curvature: Equilibration factor for curvature penalization
        :param: gradient_regularization: True --> Apply Gradient regularization depending on degree of coefficient
        :return: Loss value at the end of optimization        
        """
        self.optimizer = optimizer
        self.epochs = n_epochs

        if 'factor_ck_pressure' in kwargs:
            self.factor_ck_pressure = kwargs['factor_ck_pressure']
        if 'factor_approximation_quality' in kwargs:
            self.factor_approximation_quality = kwargs['factor_approximation_quality']
        if 'factor_curvature' in kwargs:
            self.factor_curvature = kwargs['factor_curvature']
        if 'gradient_regularization' in kwargs:
            self.gradient_regularization = kwargs['gradient_regularization']
        else:
            self.gradient_regularization = False

        if not self.performedfit:
            raise Exception("No spline data - Perform fit() first")

        if not optimizer:
            raise Exception("No optimizer specified")

        start_time = time.time()
        self._optimize_spline()

        if self.verbose:
            print("Fitting took %s seconds" % (time.time() - start_time))

        if self.continuity:
            self._establish_continuity()

        return self.total_loss_values[-1]
