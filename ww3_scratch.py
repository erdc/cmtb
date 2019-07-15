from prepdata import inputOutput
from getdatatestbed import getDataFRF
import matplotlib.pyplot as plt
import datetime as DT
from matplotlib import tri
import numpy as np
from prepdata import prepDataLib
import os, sys

start = DT.datetime(2018, 3, 2, 0, 0, 0)
end = DT.datetime(2018, 3, 7, 23, 0, 0)
## set based on model
pFlag = False
full = False
version_prefix = 'Base'
baseGridFname = '/home/spike/cmtb/grids/ww3/Mesh_interp_Mar2018.msh'
model = 'WW3'
date_str = start.strftime('%Y-%m-%dT%H%M%SZ')
path_prefix = "/home/spike/cmtb/data/{}/{}".format(model, version_prefix)
############################# initalize ##################
ww3io = inputOutput.ww3IO('')
pdl = prepDataLib.PrepDataTools()
gdtb = getDataFRF.getDataTestBed(start, end, THREDDS='FRF')
go = getDataFRF.getObs(start, end)
############ bathy #########################################
gridNodes = ww3io.load_msh(baseGridFname)       # load old bathy
bathy = gdtb.getBathyIntegratedTransect()       # get new bathy
# do interpolation
gridNodes = pdl.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, plotFname=os.path.join(path_prefix,date_str, 'bathy'+date_str))
###################### waves ################################
rawspec = go.getWaveSpec()

wavepacket = pdl.prep_spec(rawspec, version_prefix, datestr=date_str, plot=pFlag, full=full, deltaangle=5,
                                    outputPath=path_prefix, model= model)  # 50 freq bands are max for model

ww3io.writeWW3_spec(wavepacket)

from prepdata import inputOutput

########### run model ##########
data=rawspec
newfreq = [0.038]
bw = [0.038/2]
low = [0.0]
high = []
for ii in range(1, 35):
    newfreq.append(newfreq[ii - 1] * 1.1)
    low.append(newfreq[ii - 1] + (newfreq[ii] - newfreq[ii - 1]) / 2.0)
    high.append(newfreq[ii] - (newfreq[ii] - newfreq[ii - 1]) / 2.0)
    bw.append(newfreq[ii] - newfreq[ii - 1])

high.append(np.ceil(newfreq[-1]))
newfreq = np.array(newfreq)
newdir = np.arange(0, 360, 10)

blah = np.zeros((len(data['time']), 35, 72))
blah2 = np.zeros((len(data['time']), 35, 36))
for itime in range(len(data['time'])):
    for idir in range(len(data['wavedirbin'])):
        temp = data['dWED'][itime, :, idir]
        blah[itime, :, idir] = np.interp(newfreq, data['wavefreqbin'], temp)
    for ifre in range(len(newfreq)):
        tempf = blah[itime, ifre, :]
        blah2[itime, ifre, :] = np.interp(newdir, data['wavedirbin'], tempf)

# plt.pcolormesh(newfreq, newdir, data['dWED'])

print(data['wavedirbin'].shape, newfreq.shape, blah.shape, blah2.shape)

efth = np.expand_dims(blah, 1)
efth2t = np.expand_dims(blah2, 1)

# rotate directions
tt = newdir + 180.0
tt[tt >= 360.0] = tt[tt >= 360.0] - 360.0
print(tt)
ttfl = np.flipud(tt)
print(ttfl)
iddt = np.where(ttfl == 90)
idd1 = np.arange(iddt[0], len(ttfl))
idd2 = np.arange(0, iddt[0])
dirindx = np.concatenate((idd1, idd2))
print(dirindx, ttfl[dirindx])

efthtemp = efth2t[:, :, :, ::-1]
efth2 = efthtemp[:, :, :, dirindx]

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
    