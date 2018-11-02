Python Version 2.7 compatible packages were used

Necessary packages:
os	(usually comes installed with python)
scipy  	(usually comes installed with python)
numpy  	(usually comes installed with python)
pyproj 	(obtained precompiled binary from http://www.lfd.uci.edu/~gohlke/pythonlibs/pyproj-1.9.4dev.win-amd64-py2.7.exe)
	Also available via pip install (eg. pip install pyproj)
liblas 	(obtained precompiled binary from http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy libLAS-1.7.0.win-amd64-py2.7.exe)
	Also available via pip install (eg. pip install liblas)

Description:
This collection of codes for python implementation of the scale controlled interpolation methods described in:

Plant, Nathaniel G., K. Todd Holland, and Jack A. Puleo. "Analysis of the scale of errors in nearshore bathymetric data." 
           Marine Geology 191.1 (2002): 71-86.

Curerently the code package includes:

run_DEM_generator.py:	Wrapper routine to implement construction of a DEM (Digital Elevation Model).
	Reguires user inputs including:
		toolkitpath:		Path to the interpolation toolkit codes
		savepath:		Path to the final output directory for saving the DEM
		datapath:		Path to the raw data files
		datatype:		Type of data to be analyzed (file extension; e.g. 'las' for lidar tile files)
		x0:			Minimum x-value of the grid
		x1:			Maximum x-value of the grid
		y0: 			Minimum y-value of the grid
		y1:			Maximum y-value of the grid
		lambdaX:		Grid spacing in the x-direction
		lambdaY:		Grid spacing in the y-direction
		msmoothx:		Smoothing length scale in the x-direction
		msmoothy:		Smoothing length scale in the y-direction
		msmootht:		Smoothing length scale in time
		filtername:		Name of the filter type to smooth the data
		nmseitol:		Normalized error tolerance the user will tolerate in the final grid
		                           (0 - (no error) to 1 (no removal of bad points))
		grid_coord_check:	'LL' or 'UTM' - Designates if the grid supplied by the user (if one exists)
		                           is in UTM or lat-lon coordinates
		grid_filename:          Name of the grid file (if supplied)
		data_coord_check:	'LL' or 'UTM' - Designates if the data supplied by the user 
		                           is in UTM or lat-lon coordinates


dataBuilder.py:			Formats raw data into the input structure required by the interpolation routine.
gridBuilder.py:			Builds a rectagular grid using four corner points (x0,x1,y0,y1) and spacing in the 
                                x and y diregions (lambdaX, lambdaY)
scalecInterpolation.py:		Scale controlled interpolation routines	
supportingMethods.py		Collection of methods called by the interpolation routines (necessary for constructing 
                                smoothing windows, computing regressions, etc.)