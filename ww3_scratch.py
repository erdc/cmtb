from prepdata import inputOutput
from getdatatestbed import getDataFRF
import matplotlib.pyplot as plt
import datetime as DT
from matplotlib import tri
import numpy as np
from prepdata import prepDataLib
import os, sys
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
start = DT.datetime(2018, 3, 2, 0, 0, 0)
end = DT.datetime(2018, 3, 7, 23, 0, 0)
## set based on model
pFlag = True
full = True
version_prefix = 'base'
baseGridFname = '/home/spike/cmtb/grids/ww3/Mesh_interp_Mar2018.msh'
model = 'WW3'
date_str = start.strftime('%Y-%m-%dT%H%M%SZ')
path_prefix = "/home/spike/cmtb/data/{}/{}".format(model, version_prefix)
############################# initalize ##################
ww3io = inputOutput.ww3IO('')
pdl = prepDataLib.PrepDataTools()
gdtb = getDataFRF.getDataTestBed(start, end, THREDDS='CHL')
go = getDataFRF.getObs(start, end)
# ############ bathy #########################################
gridNodes = ww3io.load_msh(baseGridFname)       # load old bathy
bathy = gdtb.getBathyIntegratedTransect()       # get new bathy
# do interpolation
gridNodes = pdl.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, plotFname=os.path.join(path_prefix,date_str, 'bathy'+date_str))
###################### waves ################################
# rawspec = go.getWaveSpec()
#
# wavepacket = pdl.prep_spec(rawspec, version_prefix, datestr=date_str, plot=pFlag, full=full, deltaangle=5,
#                                     outputPath=path_prefix, model= model)  # 50 freq bands are max for model
# rawWL = go.getWL()
# ww3io.WL = pdl.prep_WL(rawWL, wavepacket['epochtime'])
#
#
# from prepdata import inputOutput
# ww3io.writeWW3_spec(wavepacket)

#
# ########### run model ##########
# dDir = 5
# rawspec=rawspec
# newfreq = [0.038]
# bw = [0.038/2]
# low = [0.0]
# high = []
# for ii in range(1, 35):
#     newfreq.append(newfreq[ii - 1] * 1.1)
#     low.append(newfreq[ii - 1] + (newfreq[ii] - newfreq[ii - 1]) / 2.0)
#     high.append(newfreq[ii] - (newfreq[ii] - newfreq[ii - 1]) / 2.0)
#     bw.append(newfreq[ii] - newfreq[ii - 1])
#
# high.append(np.ceil(newfreq[-1]))
# newfreq = np.array(newfreq)
# newdir = np.arange(0, 360, dDir)
#
# blah = np.zeros((len(rawspec['time']), 35, len(rawspec['wavedirbin'])))
# output2Dspec_tmp = np.zeros((len(rawspec['time']), 35, int(360 / dDir)))
# for itime in range(len(rawspec['time'])):
#     for idir in range(len(rawspec['wavedirbin'])):
#         temp = rawspec['dWED'][itime, :, idir]
#         blah[itime, :, idir] = np.interp(newfreq, rawspec['wavefreqbin'], temp)
#     for ifre in range(len(newfreq)):
#         tempf = blah[itime, ifre, :]
#         output2Dspec_tmp[itime, ifre, :] = np.interp(newdir, rawspec['wavedirbin'], tempf)
# ########## plot post interp
# tidx  = 100
# plt.figure()
# plt.subplot(121)
# plt.pcolormesh(output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.subplot(122)
# plt.pcolormesh(newdir, newfreq, output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.text(200, 0.5, '$E_T$ - {0:.2f}'.format(output2Dspec_tmp[tidx].sum()))
# ############# try with interp2d
# output2Dspec_tmp = np.zeros((len(rawspec['time']), 35, int(360 / dDir)))
# for itime in range(len(rawspec['time'])):
#     f = interp2d( rawspec['wavedirbin'],rawspec['wavefreqbin'], rawspec['dWED'][itime])
#     output2Dspec_tmp[itime] = f( newdir, newfreq)
# ###################
# # replot
# tidx  = 100
# plt.figure()
# plt.subplot(121)
# plt.pcolormesh(output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.subplot(122)
# plt.pcolormesh(newdir, newfreq, output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.text(200, 0.5, '$E_T$:{:.2f}'.format(output2Dspec_tmp[tidx].sum()))
#
# print(rawspec['wavedirbin'].shape, newfreq.shape, blah.shape, output2Dspec_tmp.shape)
###################################
# efth = np.expand_dims(blah, 1)
# output2Dspec_nc = np.expand_dims(output2Dspec_tmp, 1)
#
# # rotate directions
# dir_ocean = newdir + 180.0
# dir_ocean[dir_ocean >= 360.0] = dir_ocean[dir_ocean >= 360.0] - 360.0
# # print(dir_ocean)
# dir_oceanFlip = np.flipud(dir_ocean)
# # print(dir_oceanFlip)
# iddt = np.where(dir_oceanFlip == 90)
# idd1 = np.arange(iddt[0], len(dir_oceanFlip))
# idd2 = np.arange(0, iddt[0])
# dirindx = np.concatenate((idd1, idd2))
# # print(dirindx, dir_oceanFlip[dirindx])
#
# efthtemp = output2Dspec_nc[:, :, :, ::-1]
# efth2 = efthtemp[:, :, :, dirindx]
#
# ## plot finished product
# tidx  = 100
# plt.figure();
# plt.subplot(121);plt.title('WW3 Scratch: intial Product idx{}'.format(tidx))
# plt.pcolormesh(rawspec['wavedirbin'], rawspec['wavefreqbin'], rawspec['dWED'][tidx], norm=LogNorm()); plt.colorbar()
# plt.text(200, 0.25, '$E_T$:{:.2f}'.format(rawspec['dWED'][tidx].sum()))
# plt.subplot(122);plt.title('WW3 Scratch: final Product idx{}'.format(tidx))
# plt.pcolormesh(newdir, newfreq, efth2[tidx,0], norm=LogNorm()); plt.colorbar()
# plt.text(200, 0.5, '$E_T$:{:.2f}'.format(output2Dspec_tmp[tidx].sum()))
# plt.tight_layout()
############ finished ... write to file
###################################################################################
###################################################################################
# ########### start over
########### inputs ##########
dDir = 5

######################
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
newdir = np.arange(0, 360, dDir)

#################  Now try flipping order of operation, do directions then interpolation
newdir = rawspec['wavedirbin']
inputSpec = rawspec['dWED']
# rotate directions
dir_ocean = newdir + 180.0
dir_ocean[dir_ocean >= 360.0] = dir_ocean[dir_ocean >= 360.0] - 360.0
# print(dir_ocean)
dir_oceanFlip = np.flipud(dir_ocean)
# print(dir_oceanFlip)
iddt = np.where(dir_oceanFlip == 90)
idd1 = np.arange(iddt[0], len(dir_oceanFlip))
idd2 = np.arange(0, iddt[0])
dirindx = np.concatenate((idd1, idd2))
# print(dirindx, dir_oceanFlip[dirindx])
## now handle spectra like directions
efthtemp = inputSpec[:, :, ::-1]
efth2 = efthtemp[:, :, dirindx]
dirs_out = dir_oceanFlip[dirindx]
plt.figure(); tidx=100; plt.suptitle('ww3: interpIn')
plt.subplot(121)
plt.pcolormesh(rawspec['dWED'][tidx, :, dirindx].T, norm=LogNorm()); plt.colorbar()
plt.subplot(122)
plt.pcolormesh(dirs_out, rawspec['wavefreqbin'], rawspec['dWED'][tidx, :, dirindx].T, norm=LogNorm());
plt.colorbar();plt.text(200, 0.25, '$E_T$:{:.2f}'.format(rawspec['dWED'][tidx, :, dirindx].sum()))
############# try with interp2d
output2Dspec_tmp = np.zeros((len(rawspec['time']), 35, int(360 / dDir)))
for itime in range(len(rawspec['time'])):
    f = interp2d(dirs_out, rawspec['wavefreqbin'], rawspec['dWED'][itime, :, dirindx].T)
    output2Dspec_tmp[itime] = f(dirs_out, newfreq)
###################
# replot
# tidx  = 100
# plt.figure()
# plt.subplot(121)
# plt.pcolormesh(output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.subplot(122)
# plt.pcolormesh(dirs_out, newfreq, output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
# plt.text(200, 0.5, '$E_T$:{:.2f}'.format(output2Dspec_tmp[tidx].sum()))
###################################################################################
tidx  = 100
plt.figure();
plt.subplot(121);plt.title('WW3 Scratch: intial Product idx{}'.format(tidx))
plt.pcolormesh(rawspec['wavedirbin'], rawspec['wavefreqbin'], rawspec['dWED'][tidx], norm=LogNorm()); plt.colorbar()
plt.text(200, 0.25, '$E_T$:{:.2f}'.format(rawspec['dWED'][tidx].sum()))
plt.subplot(122);plt.title('WW3 Scratch: final Product idx{}'.format(tidx))
plt.pcolormesh(dirs_out, newfreq, output2Dspec_tmp[tidx], norm=LogNorm()); plt.colorbar()
plt.text(200, 0.5, '$E_T$:{:.2f}'.format(output2Dspec_tmp[tidx].sum()))
plt.tight_layout()
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
WL = go.getWL()
wind=go.getWind()
