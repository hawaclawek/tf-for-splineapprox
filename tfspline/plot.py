 #!/usr/bin/env python

"""plot.py: Helper functions to plot spline data and training loss.  
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import math
import statistics


__author__ = "Hannes Waclawek"
__version__ = "2.0"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


def plot_spline(spline, plot_input=True, deriv=0, plot_overlapping_segments = False,
                plot_corrective_polynomials = False, plot_max_h_lines=False, title_max_curvature=False,
                segment_resolution = 100, title='', ax=None):
    """Plot spline segments and input data points.  
      
    plot_input: Input data points are plotted along with the spline  
    plot_overlapping_segments: Plot polynomial pieces according to fit(segment_overlap=0.4) parameter  
    segment_resolution: Scale resolution of every polynomial segment to this number of data points.  
                        If 0, resolution of original data will be used.  
    returns: figure instance  
    """
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(7)
        fig.set_figheight(5)
    
    if title:
        ax.set_title(title)

    if plot_input and deriv == 0:
        ax.plot(spline.data_x,spline.data_y,'k.')

    colors = ['r', 'g', 'b', 'y']
    k = 0

    if not plot_overlapping_segments:
        for i in range(spline.polynum):
            x = np.linspace(spline.boundary_points[i], spline.boundary_points[i+1], round(segment_resolution))
            y = spline._evaluate_polynomial_at_x(i, deriv, x, spline._polynomial_center(i))
            ax.plot(x, y, colors[k])

            if k >= (len(colors)):
                k = 0

            ax.plot(x, y, colors[k])

            if plot_corrective_polynomials:
                if i != spline.polynum - 1:
                    ax.vlines(spline.data_x_split[i][-1], linestyles='dashed', color='k', ymin=min(y), ymax=max(y))
                if len(spline.corr_coeffs) > 0:
                    ax.plot(x, poly.polyval(x, spline.corr_coeffs[i]))
            k += 1
    else:
        for i in range(spline.polynum):
            x = np.linspace(spline.data_x_split_overlap[i][0], spline.data_x_split_overlap[i][-1], round(segment_resolution))
            y = spline._evaluate_polynomial_at_x(i, deriv, x, spline._polynomial_center(i))
            ax.plot(x, y, colors[k])

            if k >= (len(colors)):
                k = 0

            ax.plot(x, y, colors[k])
            k += 1
    
    if plot_max_h_lines:
        y = spline.evaluate_spline_at_x(spline.data_x, deriv=deriv)
        ax.hlines(max(y), spline.data_x[0], spline.data_x[-1], linestyles="dashed")
        ax.hlines(min(y), spline.data_x[0], spline.data_x[-1], linestyles="dashed")
        ax.set_yticks(np.linspace(min(y), max(y),10))

    if title_max_curvature:
        total_curvature = math.sqrt(spline.integrate_squared_spline_acceleration()) # Sqrt because better to interpret for reader - ~ RMS
        curv = "{:.2f}".format(total_curvature)
        ax.set_title(f'{title}\n\"total curvature\": {curv}') 


def plot_loss(spline, type='total', title='', ax=None):
    """Plot loss over epochs

    type='total':
        Plot combined loss
    type='curvature'
        Plot loss for integrate_squared_spline_acceleration() only
    type='ck-D'
        Plot loss for total ck_pressure() only
    type='ck-d'
        Plot loss for derivative specific ck_pressure()
    type='approx'
        Plot loss for sum of squared approximation errors
    returns: figure instance
    """
    if ax is None and type != 'all' and type != 'ck-d':
        fig, ax = plt.subplots()

    if isinstance(spline.total_loss_values, list):
        if not spline.total_loss_values:
            raise Exception("No spline loss values found. Perform optimization first.")
    else:  
        if not spline.total_loss_values.any():
            raise Exception("No spline loss values found. Perform optimization first.")

    if type == 'total':
        ax.semilogy(np.linspace(0, len(spline.total_loss_values), len(spline.total_loss_values)), spline.total_loss_values)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Total loss training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('total loss')

    elif type =='curvature':
        ax.semilogy(np.linspace(0, len(spline.total_loss_values), len(spline.total_loss_values)), spline.I_loss_values)
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Curvature training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('curvature loss')

    elif type=='ck-D':
        ax.semilogy(np.linspace(0, len(spline.total_loss_values), len(spline.total_loss_values)), spline.D_loss_values)
        ax.set_yscale('log')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('"Ck-pressure" training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss total "Ck-pressure"')

    elif type=='approx':
        ax.semilogy(np.linspace(0, len(spline.total_loss_values), len(spline.total_loss_values)), spline.e_loss_values)
        ax.set_yscale('log')
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Sum of squared errors (deriv 0) training learning curve')
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')

    elif type=='ck-d':
        fig, ax = plt.subplots(spline.ck + 1)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle('Derivative-specific "Ck-pressure" training learning curve')
        fig.text(0.5, -0.07, 'epochs', ha='center')
        fig.text(-0.07, 0.5, 'loss "Ck-pressure" derivative 0 (top) to derivative k (bottom)', va='center', rotation='vertical')
        fig.tight_layout()

        i = 0

        for e in ax:
            e.semilogy(np.linspace(0, len(spline.total_loss_values), len(spline.total_loss_values)), [row[i] for row in spline.d_loss_values])
            i += 1


def plot_spline_comparison(spline_1, spline_2, xss, spline_1_name='Spline 1',
                           spline_2_name = 'Spline 2', title = 'Title'):
    '''Plots comparison of two spline instances'''
    x = spline_1.data_x
    y = spline_1.data_y

    ck_pressure_sgd, _ = spline_1.ck_pressure()
    ck_pressure_ams, _ = spline_2.ck_pressure()

    fig, fig_axes = plt.subplots(1, 4, constrained_layout=True)
    fig.set_figwidth(24)
    fig.set_figheight(5)
    fig.suptitle(title + "\n" + spline_1_name + ", loss=%.4g" % spline_1.total_loss_values[-1]
                 + ", Ck-pressure=%.4g" % ck_pressure_sgd + "\n" + spline_2_name
                 + ", loss=%.4g" % spline_2.total_loss_values[-1] + ", Ck-pressure=%.4g" % ck_pressure_ams)

    fig_axes[0].plot(x, y, '.', c="black")

    for i in range(3):
        fig_axes[i].plot(xss, spline_1.evaluate_spline_at_x(xss, deriv=i), label=spline_1_name)
        fig_axes[i].plot(xss, spline_2.evaluate_spline_at_x(xss, deriv=i), label=spline_2_name)
        fig_axes[i].set_title(f'derivative {i}')
        fig_axes[i].legend(loc="best")

    fig_axes[-1].semilogy(spline_1.total_loss_values, label=spline_1_name)
    fig_axes[-1].semilogy(spline_2.total_loss_values, label=spline_2_name)
    fig_axes[-1].legend(loc="best")
    fig_axes[-1].set_title("Loss")
    fig_axes[-1].set_xlabel("Epochs")

    return fig