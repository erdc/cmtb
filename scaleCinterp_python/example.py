from .DEM_generator import DEM_generator
import netCDF4 as nc
import numpy as np
import os
"""
This file will give an example of how to run DEM generator with given files both a grid and a transect file
 to use either comment the other
"""
def main():
    """ """
    # gridded bathymetry data run through scaleC
    # filelist = ['http://134.164.129.55/thredds/dodsC/FRF/survey/gridded/FRF_20160726_1121_FRF_NAVD88_LARC_GPS_UTC_v20170320_grid_latlon.nc']
    # transect bathymetry data run though scaleC
    # filelist = ['http://bones/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/FRF_20160527_1119_FRF_NAVD88_LARC_GPS_UTC_v20170320.nc']
    filelist = ['http://bones/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/FRF_20160628_1120_FRF_NAVD88_LARC_GPS_UTC_v20170320.nc']
    filelist = ['http://bones/thredds/dodsC/FRF/geomorphology/elevationTransects/survey/FRF_20160726_1121_FRF_NAVD88_LARC_GPS_UTC_v20170320.nc']
    ncfile = nc.Dataset(filelist[0])  # open netCDF file -bathy = nc.Dataset(filelist[0])
    xV = ncfile['xFRF'][:].flatten()
    yV = ncfile['yFRF'][:].flatten()
    if len(ncfile['elevation'].shape) > 1:  # ndim doesn't work here
        dataX, dataY = np.meshgrid(xV, yV)
        dataX = dataX.flatten()
        dataY = dataY.flatten()
        dataZ = ncfile['elevation'][:].flatten()
    else:
        dataX = xV
        dataY = yV
        dataZ = ncfile['elevation'][:]

    x0, y0 = 1600, 1200                                 # north east corner of grid
    x1, y1 = 55, -200                                  # south west corner of grid
    dict = {'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'lambdaX': 5,                              # grid spacing in x  -  Here is where CMS would hand array of variable grid spacing
            'lambdaY': 5,                              # grid spacing in y
            'msmoothx': 40,                            # smoothing length scale in x
            'msmoothy': 100,                           # smoothing length scale in y
            'msmootht': 1,                    # smoothing length scale in Time
            'filterName': 'hanning',
            'nmseitol': 0.75,
            'grid_coord_check': 'FRF',
            'grid_filename': '',  # should be none if creating background Grid!  becomes best guess grid
            'data_coord_check': 'FRF',
            'xFRF_s': dataX,
            'yFRF_s': dataY,
            'Z_s': dataZ,
           }

    out = DEM_generator(dict)

    from matplotlib import pyplot as plt
    fig_loc = 'figures/'
    fig_name = 'survey_%s_smoothing_%dx%d_resolution_%dx%d_orgin%d%d.png' %(nc.num2date(ncfile['time'][0], ncfile['time'].units).strftime('%Y%m%d'),
                                                    dict['msmoothx'], dict['msmoothy'], dict['lambdaX'], dict['lambdaY'], dict['x0'], dict['y0'])
    loc = 800
    offshoreDepth = np.mean(out['Zi'][:,np.argmin(np.abs(out['x_out'] - loc))])

    plt.figure()
    plt.subplot(221)
    plt.title('Zi')
    plt.pcolor(out['x_out'], out['y_out'], out['Zi'] )
    plt.text(out['x_out'][len(out['x_out'])*2/3], np.median(out['y_out']), 'depth\n%.2f' %offshoreDepth, color='white' )
    plt.plot(x0, y0, 'rd')
    plt.plot(x1, y1, 'rd')
    plt.colorbar()
    plt.subplot(222)
    plt.title('MSEi')
    plt.pcolor(out['x_out'], out['y_out'], out['MSEi'])
    plt.colorbar()
    plt.subplot(223)
    plt.title('NMSEi')
    plt.pcolor(out['x_out'], out['y_out'], out['NMSEi'])
    plt.colorbar()
    plt.subplot(224)
    plt.title('MSRi')
    plt.pcolor(out['x_out'], out['y_out'], out['MSRi'])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_loc, fig_name))
    plt.close()

if __name__ == '__main__':
    print("This only executes when %s is executed rather than imported" % __file__)

