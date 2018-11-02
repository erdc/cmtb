# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:32:36 2014

@author: szthompson
"""

import scipy.io as sio
import numpy as np
import netCDF4 as nc
import pyproj

""" BUILD #1: """ 
def readInDataSet(filename):
    """This function opens a file of various types and puts data into 3 columar variables

    Args:
      filename: return: x, y, z (1 d array of observation data)

    Returns:
      x, y, z (1 d array of observation data)

    """
    dataX, dataY, dataZ = [], [], []              
    # Handle NetCDF files
    if (filename.endswith('nc')) or (filename.endswith('ncml')):
        try:
            ncfile = nc.Dataset(filename)
            dataX = ncfile['xFRF'][:]
            dataY = ncfile['yFRF'][:]
            dataZ = ncfile['elevation'][:]
        except IndexError:    # assume FRF ncfile keys
            dataX = ncfile['longitude'][:]
            dataY = ncfile['latitude'][:]
            dataZ = ncfile['elevation'][:]
        except IOError:
            print('1 - cannot open', filename, 'it may not exist.  Please check path') 
    
    # Handle LAZ files

    elif filename.endswith('.laz'):
        print('is a LAZ file, this is not compatible now')
        # import laszip 
    # Handle LAS files
    elif filename.endswith('.las'):
        from liblas import file
        try:
            f = file.File(filename, mode = 'r')
            print('Reading LAS')
            for p in f:
                if p.classification == 2:
                    dataX.append(p.x) 
                    dataY.append(p.y)
                    dataZ.append(p.z)
        except IOError: 
            print(('2 - cannot open', filename, 'it may not exist.  Please check path'))
        
    elif filename.endswith('.xyz'):
        # open file, read lines
        f = open(filename)
        lines = f.readlines()
        f.close()
        x, y, z, = [], [], []
        # parse x y z of the lines
        for line in lines:
            split = line.split()
            dataX.append(float(split[0]))
            dataY.append(float(split[1]))
            dataZ.append(float(split[2]))

    # Handle Ascii files
    elif filename.endswith('.txt'):
        # Modified from... http://stackoverflow.com/questions/16155494/python-parse-lines-of-input-file
        try:
            with open(filename, 'r') as f:
                for line in f: # Parse the columnated data on each line
                    if line.find(" "): # Each data value on each line is seperated by a space        
                        info = line.split() # Split the data into variables based on seperation criteria: the space
                        #print info[0], info[1], info[2]
                        dataX.append(float(info[0]))
                        dataY.append(float(info[1]))
                        dataZ.append(float(info[2]))
        except IOError: 
            print('3 - cannot open', filename, 'it may not exist.  Please check path')
    
    # Handle Mat files
    elif filename.endswith('.mat'):
        try:
            matFile = sio.loadmat(filename)
            #dataX = matFile['lidar']['E'][0][0]
            #dataY = matFile['lidar']['N'][0][0]
            #dataZ = matFile['lidar']['Z'][0][0]
            dataX = matFile['xi'][:,0]
            dataY = matFile['xi'][:,1]
            dataZ = matFile['xi'][:,2]
        except IOError: 
            print('4 - cannot open', filename, 'it may not exist.  Please check path')
    else:
        print('The file extension of,', filename,', is not supported. Please try again')
        print('Supported file extension: \n.nc\n.laz\n.las\n.txt\n.mat')
        
    try:  # see if its profile data
        # Reshape the data... for compatibility
        dataX = np.reshape(dataX, (len(dataX),))
        dataY = np.reshape(dataY, (len(dataY),))
        dataZ = np.reshape(dataZ, (len(dataZ),))
    except ValueError:  # this is gridded data
        xx, yy = np.meshgrid(dataX, dataY)
        dataX = np.reshape(xx, (np.size(xx),))
        dataY = np.reshape(yy, (np.size(yy),))
        dataZ = np.reshape(dataZ, (np.size(dataZ),))

    # Need to handle any NaNs?
    
    return dataX, dataY, dataZ
  
def dataBuilder(filelist, data_coord_check, EPSG=26918):
    """this function reads the measured data sets and converts to UTM (assumed Longitude, latitude)
    
    This function assumes UTM zone 18N at the FRF in NAD83

    Args:
      filelist: list of files that has single time step of bathymetry measurements
      data_coord_check: param EPSG:  EPSG code used for tranlating Latitude/longitude to UTM ( assumed FRF) UTM zone 18 N
      EPSG: Default value = 26918)

    Returns:

    """
    tempX, tempY, tempZ = [], [], [] 
    for files in filelist:
        dataX, dataY, dataZ = readInDataSet(files)
        tempX = np.concatenate((tempX,dataX))
        tempY = np.concatenate((tempY,dataY))
        tempZ = np.concatenate((tempZ,dataZ))
    
    x = np.array([tempX, tempY, np.zeros(tempX.size)]).T
    z = tempZ[:,np.newaxis]
    
    if (data_coord_check == 'LL'):
        UTM16N=pyproj.Proj("+init=EPSG:%s" % EPSG )
        [xutm,yutm] = UTM16N(tempX, tempY) # convert to UTM coord
        x = np.array([xutm, yutm, np.zeros(xutm.size)]).T

    return x, z

# load NOAA DEM
def loadNOAAdem(filename, x0, x1, y0, y1):
    """

    Args:
      filename: param x0:
      x1: param y0:
      y1: 
      x0: 
      y0: 

    Returns:

    """
    from scipy.io import netcdf 
    f = netcdf.netcdf_file(filename, 'r')
    xtmp = f.variables['x'][:]
    ytmp = f.variables['y'][:]
    ztmp = f.variables['z'][:]
    
    [xtmp,ytmp] = np.meshgrid(xtmp,ytmp)
    xtmp = xtmp.flatten(1) 
    ytmp = ytmp.flatten(1) 
    ztmp = ztmp.flatten(1) 

    Xprior = xtmp[np.where((xtmp > x0) & (xtmp < x1) & (ytmp > y0) & (ytmp < y1))]
    Yprior = ytmp[np.where((xtmp > x0) & (xtmp < x1) & (ytmp > y0) & (ytmp < y1))]
    Zprior = ztmp[np.where((xtmp > x0) & (xtmp < x1) & (ytmp > y0) & (ytmp < y1))]
    
    import pyproj
    UTM16N=pyproj.Proj("+init=EPSG:32616") # UTM coords, zone 16N, WGS84 datum
    [Xprior, Yprior] = UTM16N(Xprior,Yprior)
    
    return Xprior, Yprior, Zprior


# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:25:54 2014

@author: jwlong

# this was brought in from another file, - SB
"""
def gridBuilder(x0, x1, y0, y1, dx, dy, grid_coord_check, grid_filename, EPSG=26918):
    """uses default to EPSG code of US EST coast (FRF location),
    This function could also use key word generation
    
    Builds the grid nodes, using verticies location

    Args:
      x0: xbound of grid points
      x1: xbound of grid points
      y0: ybound of grid points
      y1: ybound of grid points
      dx: cell size in x
      dy: cell size in y
      grid_coord_check: a key to interpret the x,y bounds
      grid_filename: a place to grab previous grid points, netCDF file must have keys of xFRF, yFRF
      EPSG: return: an x grid and a y grid (Default value = 26918)

    Returns:
      an x grid and a y grid

    """
    if (grid_filename.strip() == ''):  # if there's no grid filename
        # build grid in UTM using dx dy, and 2 corners of grid(x0, y0)
        if (grid_coord_check.strip() == 'LL'):
            # must convert to UTM (meters)
            UTM16N = pyproj.Proj("+init=EPSG:%d" % EPSG)
            x0, y0 = UTM16N(x0, y0)
            x1, y1 = UTM16N(x1, y1)

        x0 = np.round(x0, decimals=0)
        x1 = np.round(x1, decimals=0)
        y0 = np.round(y0, decimals=0)
        y1 = np.round(y1, decimals=0)
        print('Generating Grid, Rounding grid nodes to whole number')
        numGridPointsX = np.ceil(np.abs(x1-x0)/dx)  # this assumes finite difference grid (points are vertex located) - sb
        numGridPointsY = np.ceil(np.abs(y1-y0)/dy)
        assert numGridPointsX > 0, 'Grid must have more than 0 nodes in X, check coordinate system'
        assert numGridPointsY > 0, 'Grid must have more than 0 nodes in Y, check coordinate system'
        # round the x min and x max to the nearest 5
        x_min = int(dx * round(float(min(x0, x1))/dx))
        x_max = int(dx * round(float(max(x0, x1))/dx))
        y_min = int(dy * round(float(min(y0, y1))/dy))
        y_max = int(dy * round(float(max(y0, y1))/dy))
        xCoord = np.arange(x_min, x_max+dx, dx)
        yCoord = np.arange(y_min, y_max+dx, dy)
        x_grid, y_grid = np.meshgrid(xCoord, yCoord)
    else:
        try:
            gridFile = nc.Dataset(grid_filename)   # load netCDF file with grid node locations
            print("here's where to get the NetCDF grid file locations %s" %(grid_filename))
            xCoord = gridFile['xFRF']
            yCoord = gridFile['yFRF']

            x_grid, y_grid = np.meshgrid(xCoord, yCoord)
        except IOError:
            print('The file,', grid_filename, ', does not exist in the path. Please try again.')

    return x_grid, y_grid