# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:33:00 2014

@author: jwlong, edited Spicer Bak 8/4/17
"""
import numpy as np
from .dataBuilder import dataBuilder, gridBuilder
from .subsampleData import subsampleData
from .scalecInterpolation import scalecInterpTilePerturbations
import datetime as DT
import os
import scipy.io as spio
from matplotlib import pyplot as plt
from scipy import signal

def DEM_generator(dict):
    """ This is a function that takes in data with the below keys, and will generate a cartesian grid if there is not

    one already built.  if there is a grid given it will grid to the same locations.  If there is not it will generate
    a new grid with the x0,y0 and x1, y1, lambdaX and lambdaY keys.  This function will import data, subsample data,
    and grid data using scaleCinterp from plant 2002.  This will not incorporate to a background grid.

    Args: dict (dict): input dictionary with below keys
        'x0':              # Minimum x-value of the output grid (origin)

        'y0':              # Minimum y-value of the output grid

        'x1':              # Maximum x-value of the output grid

        'y1':              # Maximum y-value of the output grid

        'grid_filename':   # full filepath of the existing grid (which this will build upon), if it exists

        'lambdaY':         # grid spacing in the y-direction

        'lambdaX':         # Grid spacing in the x-direction

        'msmoothx':        # Smoothing length scale in the x-direction

        'msmoothy':        # Smoothing length scale in the y-direction

        'msmootht':        # Smoothing length scale in time

        'filtername':      # Name of the filter type to smooth the data
                           #      ['hanning', 'linloess', 'quadloess', 'boxcar', si']

        'nmseitol':        # Normalized error tolerance the user will tolerate in the final grid
                           #      (0 - (no error) to 1 (no removal of bad points))

        'xFRF_s':          # survey xFRF coordinates

        'yFRF_s':          # survey yFRF coordinates

        'Z_s':             # survey bottom elevations

    Returns:
      dict with keys
        'zi': the depth estimate

        'msei': the mean square interpolation error estimate (units of z)

        'nmsei': the normalized mean square error

        'msri': the mean square residuals

    """
    x0 = dict['x0']             # Minimum x-value of the output grid (origin)
    y0 = dict['y0']             # Minimum y-value of the output grid
    x1 = dict['x1']             # Maximum x-value of the output grid
    y1 = dict['y1']             # Maximum y-value of the output grid
    lambdaY = dict['lambdaY']        # grid spacing in the y-direction
    lambdaX = dict['lambdaX']        # Grid spacing in the x-direction
    msmoothx = dict['msmoothx']      # Smoothing length scale in the x-direction
    msmoothy = dict['msmoothy']      # Smoothing length scale in the y-direction
    msmootht = dict['msmootht']      # Smoothing length scale in time
    filtername = dict['filterName']  # Name of the filter type to smooth the data
                                     #      ['hanning', 'linloess', 'quadloess', 'boxcar', si']
    nmseitol = dict['nmseitol']      # Normalized error tolerance the user will tolerate in the final grid
                                     #      (0 - (no error) to 1 (no removal of bad points))
    xFRF_s = dict['xFRF_s']          # survey xFRF coordinates
    yFRF_s = dict['yFRF_s']          # survey yFRF coordinates
    Z_s = dict['Z_s']                # survey bottom elevations

    #### data checks ###########3
    filters = ['hanning', 'linloess', 'quadloess', 'boxcar', 'si']
    assert filtername in filters, 'Check filter name, not appropriate for current DEM generator function'
    assert xFRF_s.shape == Z_s.shape and yFRF_s.shape == Z_s.shape, 'DEM GENERATOR data input data must be all the same shape, 1D linear'

    ####################################################################
    ################################# Load Data ########################
    ####################################################################
    t = DT.datetime.now()
    # I use my dictionary instead of the dataBuilder function from plant's code !!!!!
    # x, z = dataBuilder(filelist, data_coord_check='FRF')
    x = np.array([xFRF_s, yFRF_s, np.zeros(xFRF_s.size)]).T
    z = Z_s[:, np.newaxis]
    s = np.zeros((np.size(x[:,1]),1))    # TODO estimate measurement error from the crab and incorporate to scripts
    print('loading time is %s seconds' % (DT.datetime.now() - t))
    assert x.shape[0] > 1, 'Data Did not Load!'
    ####################################################################
    # Call grid builder to make a grid based on x,y min and max values #
    ####################################################################
    x_grid, y_grid = gridBuilder(x0, x1, y0, y1, lambdaX, lambdaY, dict['grid_coord_check'], dict['grid_filename'])
    t_grid = np.zeros_like((x_grid))  # Interpolate in time -- Not Developed Yet, but place holder there

    xi = np.array([x_grid.flatten(), y_grid.flatten(), t_grid.flatten()]).T  # grid locations, flatten make row-major style
    # now make smoothing array same shape as  xi
    xsm = msmoothx*np.ones_like(x_grid)
    ysm = msmoothy*np.ones_like(y_grid)
    tsm = msmootht*np.ones_like(t_grid)
    lx = np.array([xsm.flatten(), ysm.flatten(), tsm.flatten()]).T  # smoothing array , flatten takes row-major style

    # why don't we just pass the meshgrid output?  that would be simpler?
    N, M = np.shape(x_grid)
    x_out = x_grid[0,:].copy()  # grid coordinate for output
    y_out = y_grid[:,0].copy()  # grid coordinate for output

    #####################################################################
    # subsample the data   ##############################################
    #####################################################################

    DXsmooth = np.array([msmoothx, msmoothy, msmootht])/4
    DXsmooth[2] = 1  # this hard codes a time smoothing of 1 for subsampling of data
    t = DT.datetime.now()
    Xi, zprime, si = subsampleData(x, z, s, DXsmooth)
    print('subsampling time is %s seconds' % (DT.datetime.now() - t))

    # a plot to compare original data to subsampled data
    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(x[:,0], x[:,1], '.', label='Raw')
    # plt.plot(Xi[:,0], Xi[:,1], '.', label='SubSampled')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(np.sqrt(x[:, 0]**2 + x[:, 1]**2), z, '.', label='raw')
    # plt.plot( np.sqrt(Xi[:,0]**2 + Xi[:,1]**2), zprime, '.', label='subsampled')
    # plt.legend()
    # plt.close()

    ##############################################################
    # Send it all into scalecinterpolation  -  Here is where the interpolation takes place
    #####################################################################
    t = DT.datetime.now()
    zi, msei, nmsei, msri = scalecInterpTilePerturbations(Xi, zprime, si, xi, lx, filtername, nmseitol)
    print('Interpolating time is %s seconds' % (DT.datetime.now() - t))

    # reshape
    zi = np.reshape(zi, (M, N)).T           # zi, the estimate
    msei = np.reshape(msei, (M, N)).T       # msei, the mean square interpolation error estimate (units of z)
    nmsei = np.reshape(nmsei, (M, N)).T     # nmsei, the normalized mean square error
    msri = np.reshape(msri, (M, N)).T       # msri, the mean square residuals

    out = {'Zi': zi,
           'MSEi': msei,
           'NMSEi': nmsei,
           'MSRi': msri,
           'x_out': x_out,
           'y_out': y_out,
           }

    return out

def makeWBflow(y_grid, Nysmooth, lambdaY):
    """This is the weight scaling script that Nathanial and Meg use.
    Looks like it uses a Gaussian function at each edge as the weight scaling factor
        (or 1 - Gaussian to be more specific).
    It only splines in the alongshore direction, not cross-shore!

    Args:
      y_grid: 2D grid of y-coordinates (i.e., y meshgrid output)
      Nysmooth: number of smoothing nodes at each y-edge.
    This is combined with lambdaY to get basically a scaled standard deviation of a Gaussian distribution
      lambdaY: grid spacing in Y.
    This is combined with Nysmooth to get basically a scaled standard deviation of a Gaussian distribution

    Returns:
      wbflow - scaling factors for the bspline weights

    """


    # this is what Meg and Mathan have
    wbflowS1 = np.exp(-1*np.power((y_grid - y_grid[0,0])/float(1*Nysmooth*lambdaY), 2))
    wbflowS2 = wbflowS1 + np.exp(-1*np.power((y_grid - y_grid[-1,-1])/float(1*Nysmooth*lambdaY), 2))
    wbflowS3 = 1 - wbflowS2
    # if less than zero set to zero
    wbflowS3[wbflowS3<0] = 0
    wbflow = wbflowS3/np.amax(wbflowS3)

    return wbflow

def makeWBflow2D(dict):
    """This is the weight edge scaling function that I developed that works in 2D.
    It uses two 1D tukey filters combined together using an outer product to get the edge scaling.
    ax and ay are the parameters of the cross-shore and alongshore edge scaling factors.
    
    for a Tukey filter, a = 0 gives you no scaling (rectangular window), a = 1 gives you a Hann filter (spelling?)

    Args:
      dict: Keys:
    :key x_grid: 2D x-coordinate grid from Meshgrid
    :key y_grid: 2D y-coordinate grid from meshgrid
    :key ax: alpha value for the x-direction tukey filter.
    :key ay: alpha value for the y-direction tukey filter.

    Returns:
      wbflow - scaling factors for the bspline weights

    """

    x_grid = dict['x_grid']
    y_grid = dict['y_grid']
    ax = dict['ax']
    ay = dict['ay']

    # cannot exceed 1
    if ax > 1:
        ax = 1
    if ay > 1:
        ay = 1

    # we are going to do one that smoothes all around the edges.

    window_y = signal.tukey(np.shape(y_grid)[0], alpha=ay)
    window_x = signal.tukey(np.shape(x_grid)[1], alpha=ax)
    window = np.outer(window_y, window_x)

    """
    # what does this look like?
    sloc = 'C:\\Users\dyoung8\Desktop\David Stuff\Projects\CSHORE\Bathy Interpolation\WeightFilterTest'
    sname = 'wbWeights'
    # also plot the output like a boss
    plt.pcolor(x_grid, y_grid, window, cmap=plt.cm.jet, vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('Weights', fontsize=16)
    plt.xlabel('Cross-shore - $x$ ($m$)', fontsize=16)
    plt.ylabel('Alongshore - $y$ ($m$)', fontsize=16)
    plt.legend(prop={'size': 14})
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    ax1 = plt.gca()
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.axis('tight')
    plt.tight_layout()
    plt.savefig(os.path.join(sloc, sname))
    plt.close()
    """


    return window
