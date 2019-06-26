from prepdata import inputOutput
from getdatatestbed import getDataFRF
import matplotlib.pyplot as plt
import meshio as mio
import datetime as DT
from matplotlib import tri
import numpy as np

ww3io = inputOutput.ww3IO('')
start = DT.datetime(2018, 1, 2)
end = DT.datetime(2018, 2,1)

fname = '/home/spike/cmtb/grids/ww3/Mesh_interp_Mar2018.msh'
gridNodes = mio.read(fname)
gdtb = getDataFRF.getDataTestBed(start, end, THREDDS='CHL')
bathy = gdtb.getBathyIntegratedTransect()
from prepdata import prepDataLib
pdl = prepDataLib.PrepDataTools()
gridNodes = pdl.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, plotFname='myOutputplot.png')
mio.write_points_cells('myfile_updated.msh', gridNodes.points, gridNodes.cells)


go = getDataFRF.getObs(start, end)


def make_ww3spec_nc(d1, d2, fname):
    go = getDataFRF.getObs(d1, d2, THREDDS='CHL')
    data = go.getWaveSpec()

    newfreq, bw, low, high = [], [], [], []
    newfreq.append(0.038)  # beginning of new frequency bins
    bw.append(0.038 / 2)
    low.append(0.0)
    for ii in range(1, 35):
        newfreq.append(newfreq[ii - 1] * 1.1)
        low.append(newfreq[ii - 1] + (newfreq[ii] - newfreq[ii - 1]) / 2.0)
        high.append(newfreq[ii] - (newfreq[ii] - newfreq[ii - 1]) / 2.0)
        bw.append(newfreq[ii] - newfreq[ii - 1])

    high.append(np.ceil(newfreq[-1]))
    newfreq = np.array(newfreq)
    newdir = np.arange(0, 360, 10)
    print(newfreq, newdir)

###############################################################
## plot unstructured data on large grid
# cMin = np.min(-data.points[:, 2])
# cMax = np.max(-data.points[:, 2])
# plt.figure()
# plt.tripcolor(data.points[:, 0], data.points[:, 1], -data.points[:, 2])
# plt.triplot(data.points[:, 0], data.points[:, 1], 'k.', ms=2)
# plt.plot([-75.75141733166804,  -75.74613556618358], [36.18187930529555,   36.18328286289117], color='black',  ms=10) # [],'k-')
# plt.text(-75.75141733166804, 36.18187930529555, 'FRF                           ', fontsize = 14, horizontalalignment='right')
# ## data. points are lon, lat, elevations from below
#
# ##  over plot new bathy
# plt.pcolormesh(bathy['lon'], bathy['lat'], bathy['elevation'], vmin=cMin, vmax=cMax)
# plt.ylim([bathy['lat'].min(), bathy['lat'].max()])
# plt.xlim([bathy['lon'].min(), bathy['lon'].max()])
# #################################################
# ## begin interpolations
#
# #
# # fcub = tri.CubicTriInterpolator(triObj, newBathy['elevation'].flatten())
# # newZsCubic = fcub(data.points[:, 0], data.points[:, 1])
# ###################################################
# # make plot
# vmin = np.min(bathy['elevation'])
# vmax = np.max(bathy['elevation'])
# plt.figure()
# ax1 = plt.subplot(131)
# a = ax1.pcolormesh(bathy['lon'], bathy['lat'], bathy['elevation'], cmap='RdBu', vmin=vmin, vmax=vmax)
# plt.colorbar(a)
# ax1.set_title('Original -- new data -- before interp')
# ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
# b = ax2.tripcolor(data.points[:,0], data.points[:,1], newZs, cmap='RdBu', vmin=vmin, vmax=vmax)
# plt.colorbar(b)
# ax2.set_title('Newly Iterped values to triangular mesh -- linear')
# ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
# c = ax3.tripcolor(data.points[:,0], data.points[:,1], newZsCubic, cmap='RdBu', vmin=vmin, vmax=vmax)
# plt.colorbar(c)
# ax3.set_title('Newly Iterped values to triangular mesh -- cubic')


###################################################################
# replace values in original
# handle negative values here

# ########################333
# plot merged area
    