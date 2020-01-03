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
# gridNodes = ww3io.load_msh(baseGridFname)       # load old bathy
# bathy = gdtb.getBathyIntegratedTransect()       # get new bathy
# # do interpolation
# gridNodes = pdl.prep_Bathy(bathy, gridNodes=gridNodes, unstructured=True, plotFname=os.path.join(path_prefix,date_str, 'bathy'+date_str))
###################### waves ################################
rawspec = go.getWaveSpec()

wavepacket = pdl.prep_spec(rawspec, version_prefix, datestr=date_str, plot=pFlag, full=full, deltaangle=5,
                                    outputPath=path_prefix, model= model)  # 50 freq bands are max for model
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
def ww3_shel(fname, wind, WL, savepoints):

    simStart = wind['time'].strfmt('%Y%m%d %H%M%S')
    simEnd= wind['time'].strfmt('%Y%m%d %H%M%S')
    ############### begin writing
    f = open(fname, 'w')
    openingString = "$ WAVEWATCH III shell input file                                       $\n\
    $ -------------------------------------------------------------------- $\n\
    $ Define input to be used with flag for use and flag for definition\n\
    $ as a homogeneous field (first three only); seven input lines.\n\
    $ -------------------------------------------------------------------- $\n$\n"

    f.write(openingString)
    f.write()

       F F     Water levels
       F F     Currents
       F F     Winds
       F       Ice concentrations
       F       Assimilation data : Mean parameters
       F       Assimilation data : 1-D spectra
       F       Assimilation data : 2-D spectra.
    secondString = "$\n$ Time frame of calculations ----------------------------------------- $\n" \
                   "$ - Starting time in yyyymmdd hhmmss format.\n$ - Ending time in yyyymmdd hhmmss format.\n$"
    f.write(secondString)
    f.write("{}\n{}".format(simStart, simEnd))
   # 20110827 120000
   # 20110829 000000

$
$ Define output data ------------------------------------------------- $
$
$ Define output server mode. This is used only in the parallel version
$ of the model. To keep the input file consistent, it is always needed.
$ IOSTYP = 1 is generally recommended. IOSTYP > 2 may be more efficient
$ for massively parallel computations. Only IOSTYP = 0 requires a true
$ parallel file system like GPFS.
$
$    IOSTYP = 0 : No data server processes, direct access output from
$                 each process (requirese true parallel file system).
$             1 : No data server process. All output for each type
$                 performed by process that performes computations too.
$             2 : Last process is reserved for all output, and does no
$                 computing.
$             3 : Multiple dedicated output processes.
$
   3
$
$ Five output types are available (see below). All output types share
$ a similar format for the first input line:
$ - first time in yyyymmdd hhmmss format, output interval (s), and
$   last time in yyyymmdd hhmmss format (all integers).
$ Output is disabled by setting the output interval to 0.
$
$ Type 1 : Fields of mean wave parameters
$          Standard line and line with flags to activate output fields
$          as defined in section 2.4 of the manual. The second line is
$          not supplied if no output is requested.
$                               The raw data file is out_grd.ww3,
$                               see w3iogo.ftn for additional doc.
$
$
$
   20110826 000000  1800  20110829 000000
$----------------------------------------------------------------
$ Output request flags identifying fields as in ww3_shel input and
$ section 2.4 of the manual.
$
N
HS LM T02 T01 DIR UST CHA CGE DTD FC CFX CFD
$
$----------------------------------------------------------------
$
$ Type 2 : Point output
$          Standard line and a number of lines identifying the
$          longitude, latitude and name (C*10) of output points.
$          The list is closed by defining a point with the name
$          'STOPSTRING'. No point info read if no point output is
$          requested (i.e., no 'STOPSTRING' needed).
$          Example for spherical grid.
$                               The raw data file is out_pnt.ww3,
$                               see w3iogo.ftn for additional doc.
$
$   NOTE : Spaces may be included in the name, but this is not
$          advised, because it will break the GrADS utility to
$          plots spectra and source terms, and will make it more
$          difficult to use point names in data files.
$
   20110826 000000 1800  20110829 000000
$output points for cartesian grid
$output points for IROISE
$
 -75.5947        36.2605     P26
 -75.7141        36.1999     P17
 -75.7393        36.1893     P11
 -75.7433        36.1883     P08
 -75.7465        36.1873     P06
 -75.7482        36.1868     P05
 -75.7487        36.1868     P04
 -75.7498        36.1865     P03
 -75.7505        36.1863     P02
$
     0.0   0.0  'STOPSTRING'
$
$ Type 3 : Output along  track.
$          Flag for formatted input file.
$                         The data files are track_i.ww3 and
$                         track_o.ww3, see w3iotr.ftn for ad. doc.
$
   20040101 000000   1  20040101 000000
   T
$
$ Type 4 : Restart files (no additional data required).
$                               The data file is restartN.ww3, see
$                               w3iors.ftn for additional doc.
$
   20100101 000000 1728000  20100603 000000
$
$ Type 5 : Boundary data (no additional data required).
$                               The data file is nestN.ww3, see
$                               w3iobp.ftn for additional doc.
$
   20040601 000000    0  20040103 000000
$
$ Type 6 : Separated wave field data (dummy for now).
$          First, last step IX and IY, flag for formatted file
$
   20040101 000000    0  20040603 000000
$
$ Testing of output through parameter list (C/TPAR) ------------------ $
$    Time for output and field flags as in above output type 1.
$
$  19680606 014500
$    T T T T T  T T T T T  T T T T T  T
$
$ Homogeneous field data --------------------------------------------- $
$ Homogeneous fields can be defined by a list of lines containing an ID
$ string 'LEV' 'CUR' 'WND', date and time information (yyyymmdd
$ hhmmss), value (S.I. units), direction (current and wind, oceanographic
$ convention degrees)) and air-sea temparature difference (degrees C).
$ 'STP' is mandatory stop string.
$
LEV  20110826  000000  -0.323
LEV  20110826  000600  -0.359
LEV  20110826  001200  -0.372
LEV  20110826  001800  -0.401
LEV  20110826  002400  -0.453
LEV  20110826  003000  -0.457
LEV  20110826  003600  -0.510
LEV  20110826  004200  -0.534
LEV  20110826  004800  -0.557
LEV  20110826  005400  -0.579
LEV  20110826  010000  -0.622
LEV  20110826  010600  -0.622
LEV  20110826  011200  -0.624
LEV  20110826  011800  -0.665
LEV  20110826  012400  -0.702
LEV  20110826  013000  -0.736
LEV  20110826  013600  -0.752
LEV  20110826  014200  -0.786
LEV  20110826  014800  -0.808
LEV  20110826  015400  -0.810
LEV  20110826  020000  -0.858
LEV  20110826  020600  -0.856
LEV  20110826  021200  -0.876
LEV  20110826  021800  -0.894
LEV  20110826  022400  -0.905
LEV  20110826  023000  -0.890
LEV  20110826  023600  -0.927
LEV  20110826  024200  -0.957
LEV  20110826  024800  -0.943
LEV  20110826  025400  -0.907
LEV  20110826  030000  -0.880
LEV  20110826  030600  -0.880
LEV  20110826  031200  -0.892
LEV  20110826  031800  -0.894
LEV  20110826  032400  -0.916
LEV  20110826  033000  -0.888
LEV  20110826  033600  -0.897
LEV  20110826  034200  -0.922
LEV  20110826  034800  -0.940
LEV  20110826  035400  -0.950
LEV  20110826  040000  -0.933
LEV  20110826  040600  -0.958
LEV  20110826  041200  -0.951
LEV  20110826  041800  -0.973
LEV  20110826  042400  -0.965
LEV  20110826  043000  -0.929
LEV  20110826  043600  -0.940
LEV  20110826  044200  -0.936
LEV  20110826  044800  -0.938
LEV  20110826  045400  -0.907
LEV  20110826  050000  -0.877
LEV  20110826  050600  -0.877
LEV  20110826  051200  -0.828
LEV  20110826  051800  -0.814
LEV  20110826  052400  -0.809
LEV  20110826  053000  -0.746
LEV  20110826  053600  -0.727
LEV  20110826  054200  -0.741
LEV  20110826  054800  -0.744
LEV  20110826  055400  -0.731
LEV  20110826  060000  -0.682
LEV  20110826  060600  -0.668
LEV  20110826  061200  -0.645
LEV  20110826  061800  -0.617
LEV  20110826  062400  -0.574
LEV  20110826  063000  -0.559
LEV  20110826  063600  -0.539
LEV  20110826  064200  -0.491
LEV  20110826  064800  -0.473
LEV  20110826  065400  -0.449
LEV  20110826  070000  -0.413
LEV  20110826  070600  -0.391
LEV  20110826  071200  -0.365
LEV  20110826  071800  -0.341
LEV  20110826  072400  -0.354
LEV  20110826  073000  -0.318
LEV  20110826  073600  -0.288
LEV  20110826  074200  -0.240
LEV  20110826  074800  -0.224
LEV  20110826  075400  -0.235
LEV  20110826  080000  -0.216
LEV  20110826  080600  -0.198
LEV  20110826  081200  -0.181
LEV  20110826  081800  -0.172
LEV  20110826  082400  -0.153
LEV  20110826  083000  -0.155
LEV  20110826  083600  -0.149
LEV  20110826  084200  -0.153
LEV  20110826  084800  -0.166
LEV  20110826  085400  -0.190
LEV  20110826  090000  -0.166
LEV  20110826  090600  -0.147
LEV  20110826  091200  -0.159
LEV  20110826  091800  -0.140
LEV  20110826  092400  -0.138
LEV  20110826  093000  -0.132
LEV  20110826  093600  -0.147
LEV  20110826  094200  -0.176
LEV  20110826  094800  -0.172
LEV  20110826  095400  -0.162
LEV  20110826  100000  -0.145
LEV  20110826  100600  -0.136
LEV  20110826  101200  -0.162
LEV  20110826  101800  -0.151
LEV  20110826  102400  -0.186
LEV  20110826  103000  -0.167
LEV  20110826  103600  -0.156
LEV  20110826  104200  -0.196
LEV  20110826  104800  -0.207
LEV  20110826  105400  -0.200
LEV  20110826  110000  -0.211
LEV  20110826  110600  -0.182
LEV  20110826  111200  -0.214
LEV  20110826  111800  -0.211
LEV  20110826  112400  -0.229
LEV  20110826  113000  -0.263
LEV  20110826  113600  -0.265
LEV  20110826  114200  -0.282
LEV  20110826  114800  -0.289
LEV  20110826  115400  -0.330
LEV  20110826  120000  -0.347
LEV  20110826  120600  -0.338
LEV  20110826  121200  -0.368
LEV  20110826  121800  -0.385
LEV  20110826  122400  -0.406
LEV  20110826  123000  -0.441
LEV  20110826  123600  -0.483
LEV  20110826  124200  -0.537
LEV  20110826  124800  -0.578
LEV  20110826  125400  -0.562
LEV  20110826  130000  -0.626
LEV  20110826  130600  -0.635
LEV  20110826  131200  -0.647
LEV  20110826  131800  -0.662
LEV  20110826  132400  -0.683
LEV  20110826  133000  -0.665
LEV  20110826  133600  -0.715
LEV  20110826  134200  -0.746
LEV  20110826  134800  -0.755
LEV  20110826  135400  -0.785
LEV  20110826  140000  -0.798
LEV  20110826  140600  -0.793
LEV  20110826  141200  -0.809
LEV  20110826  141800  -0.848
LEV  20110826  142400  -0.855
LEV  20110826  143000  -0.880
LEV  20110826  143600  -0.882
LEV  20110826  144200  -0.909
LEV  20110826  144800  -0.938
LEV  20110826  145400  -0.925
LEV  20110826  150000  -0.932
LEV  20110826  150600  -0.911
LEV  20110826  151200  -0.945
LEV  20110826  151800  -0.953
LEV  20110826  152400  -0.935
LEV  20110826  153000  -0.937
LEV  20110826  153600  -0.902
LEV  20110826  154200  -0.915
LEV  20110826  154800  -0.910
LEV  20110826  155400  -0.879
LEV  20110826  160000  -0.930
LEV  20110826  160600  -0.910
LEV  20110826  161200  -0.887
LEV  20110826  161800  -0.899
LEV  20110826  162400  -0.856
LEV  20110826  163000  -0.833
LEV  20110826  163600  -0.809
LEV  20110826  164200  -0.816
LEV  20110826  164800  -0.814
LEV  20110826  165400  -0.816
LEV  20110826  170000  -0.819
LEV  20110826  170600  -0.785
LEV  20110826  171200  -0.758
LEV  20110826  171800  -0.735
LEV  20110826  172400  -0.724
LEV  20110826  173000  -0.679
LEV  20110826  173600  -0.656
LEV  20110826  174200  -0.639
LEV  20110826  174800  -0.596
LEV  20110826  175400  -0.578
LEV  20110826  180000  -0.558
LEV  20110826  180600  -0.535
LEV  20110826  181200  -0.512
LEV  20110826  181800  -0.484
LEV  20110826  182400  -0.441
LEV  20110826  183000  -0.417
LEV  20110826  183600  -0.395
LEV  20110826  184200  -0.387
LEV  20110826  184800  -0.335
LEV  20110826  185400  -0.298
LEV  20110826  190000  -0.280
LEV  20110826  190600  -0.262
LEV  20110826  191200  -0.230
LEV  20110826  191800  -0.206
LEV  20110826  192400  -0.166
LEV  20110826  193000  -0.140
LEV  20110826  193600  -0.107
LEV  20110826  194200  -0.098
LEV  20110826  194800  -0.080
LEV  20110826  195400  -0.040
LEV  20110826  200000  0.019
LEV  20110826  200600  0.015
LEV  20110826  201200  0.023
LEV  20110826  201800  0.041
LEV  20110826  202400  0.071
LEV  20110826  203000  0.095
LEV  20110826  203600  0.127
LEV  20110826  204200  0.141
LEV  20110826  204800  0.145
LEV  20110826  205400  0.180
LEV  20110826  210000  0.204
LEV  20110826  210600  0.198
LEV  20110826  211200  0.195
LEV  20110826  211800  0.229
LEV  20110826  212400  0.224
LEV  20110826  213000  0.226
LEV  20110826  213600  0.220
LEV  20110826  214200  0.236
LEV  20110826  214800  0.220
LEV  20110826  215400  0.233
LEV  20110826  220000  0.226
LEV  20110826  220600  0.231
LEV  20110826  221200  0.219
LEV  20110826  221800  0.210
LEV  20110826  222400  0.216
LEV  20110826  223000  0.201
LEV  20110826  223600  0.169
LEV  20110826  224200  0.175
LEV  20110826  224800  0.199
LEV  20110826  225400  0.160
LEV  20110826  230000  0.167
LEV  20110826  230600  0.123
LEV  20110826  231200  0.132
LEV  20110826  231800  0.156
LEV  20110826  232400  0.129
LEV  20110826  233000  0.089
LEV  20110826  233600  0.106
LEV  20110826  234200  0.107
LEV  20110826  234800  0.079
LEV  20110826  235400  0.037
LEV  20110827  000000  0.031
LEV  20110827  000600  -0.005
LEV  20110827  001200  -0.034
LEV  20110827  001800  -0.084
LEV  20110827  002400  -0.063
LEV  20110827  003000  -0.054
LEV  20110827  003600  -0.082
LEV  20110827  004200  -0.162
LEV  20110827  004800  -0.140
LEV  20110827  005400  -0.177
LEV  20110827  010000  -0.216
LEV  20110827  010600  -0.250
LEV  20110827  011200  -0.257
LEV  20110827  011800  -0.237
LEV  20110827  012400  -0.295
LEV  20110827  013000  -0.401
LEV  20110827  013600  -0.380
LEV  20110827  014200  -0.388
LEV  20110827  014800  -0.433
LEV  20110827  015400  -0.502
LEV  20110827  020000  -0.547
LEV  20110827  020600  -0.559
LEV  20110827  021200  -0.526
LEV  20110827  021800  -0.588
LEV  20110827  022400  -0.560
LEV  20110827  023000  -0.601
LEV  20110827  023600  -0.565
LEV  20110827  024200  -0.622
LEV  20110827  024800  -0.586
LEV  20110827  025400  -0.604
LEV  20110827  030000  -0.671
LEV  20110827  030600  -0.657
LEV  20110827  031200  -0.721
LEV  20110827  031800  -0.765
LEV  20110827  032400  -0.718
LEV  20110827  033000  -0.744
LEV  20110827  033600  -0.755
LEV  20110827  034200  -0.720
LEV  20110827  034800  -0.741
LEV  20110827  035400  -0.696
LEV  20110827  040000  -0.730
LEV  20110827  040600  -0.661
LEV  20110827  041200  -0.710
LEV  20110827  041800  -0.711
LEV  20110827  042400  -0.744
LEV  20110827  043000  -0.705
LEV  20110827  043600  -0.718
LEV  20110827  044200  -0.714
LEV  20110827  044800  -0.697
LEV  20110827  045400  -0.624
LEV  20110827  050000  -0.695
LEV  20110827  050600  -0.639
LEV  20110827  051200  -0.624
LEV  20110827  051800  -0.563
LEV  20110827  052400  -0.540
LEV  20110827  053000  -0.628
LEV  20110827  053600  -0.572
LEV  20110827  054200  -0.548
LEV  20110827  054800  -0.482
LEV  20110827  055400  -0.516
LEV  20110827  060000  -0.450
LEV  20110827  060600  -0.458
LEV  20110827  061200  -0.522
LEV  20110827  061800  -0.456
LEV  20110827  062400  -0.468
LEV  20110827  063000  -0.417
LEV  20110827  063600  -0.387
LEV  20110827  064200  -0.450
LEV  20110827  064800  -0.307
LEV  20110827  065400  -0.366
LEV  20110827  070000  -0.332
LEV  20110827  070600  -0.284
LEV  20110827  071200  -0.263
LEV  20110827  071800  -0.176
LEV  20110827  072400  -0.088
LEV  20110827  073000  -0.170
LEV  20110827  073600  -0.147
LEV  20110827  074200  -0.044
LEV  20110827  074800  0.061
LEV  20110827  075400  0.106
LEV  20110827  080000  0.017
LEV  20110827  080600  0.130
LEV  20110827  081200  0.132
LEV  20110827  081800  0.127
LEV  20110827  082400  0.162
LEV  20110827  083000  0.182
LEV  20110827  083600  0.210
LEV  20110827  084200  0.158
LEV  20110827  084800  0.257
LEV  20110827  085400  0.309
LEV  20110827  090000  0.268
LEV  20110827  090600  0.294
LEV  20110827  091200  0.373
LEV  20110827  091800  0.310
LEV  20110827  092400  0.338
LEV  20110827  093000  0.319
LEV  20110827  093600  0.408
LEV  20110827  094200  0.332
LEV  20110827  094800  0.373
LEV  20110827  095400  0.377
LEV  20110827  100000  0.428
LEV  20110827  100600  0.403
LEV  20110827  101200  0.404
LEV  20110827  101800  0.402
LEV  20110827  102400  0.499
LEV  20110827  103000  0.421
LEV  20110827  103600  0.431
LEV  20110827  104200  0.428
LEV  20110827  104800  0.507
LEV  20110827  105400  0.434
LEV  20110827  110000  0.512
LEV  20110827  110600  0.458
LEV  20110827  111200  0.388
LEV  20110827  111800  0.501
LEV  20110827  112400  0.409
LEV  20110827  113000  0.381
LEV  20110827  113600  0.431
LEV  20110827  114200  0.505
LEV  20110827  114800  0.450
LEV  20110827  115400  0.451
LEV  20110827  120000  0.418
LEV  20110827  120600  0.437
LEV  20110827  121200  0.401
LEV  20110827  121800  0.375
LEV  20110827  122400  0.389
LEV  20110827  123000  0.449
LEV  20110827  123600  0.297
LEV  20110827  124200  0.321
LEV  20110827  124800  0.299
LEV  20110827  125400  0.328
LEV  20110827  130000  0.325
LEV  20110827  130600  0.396
LEV  20110827  131200  0.398
LEV  20110827  131800  0.274
LEV  20110827  132400  0.212
LEV  20110827  133000  0.174
LEV  20110827  133600  0.198
LEV  20110827  134200  0.196
LEV  20110827  134800  0.161
LEV  20110827  135400  0.153
LEV  20110827  140000  0.161
LEV  20110827  140600  0.094
LEV  20110827  141200  0.096
LEV  20110827  141800  -0.031
LEV  20110827  142400  0.035
LEV  20110827  143000  -0.012
LEV  20110827  143600  0.050
LEV  20110827  144200  -0.144
LEV  20110827  144800  -0.132
LEV  20110827  145400  -0.174
LEV  20110827  150000  -0.217
LEV  20110827  150600  -0.189
LEV  20110827  151200  -0.330
LEV  20110827  151800  -0.212
LEV  20110827  152400  -0.306
LEV  20110827  153000  -0.318
LEV  20110827  153600  -0.296
LEV  20110827  154200  -0.302
LEV  20110827  154800  -0.240
LEV  20110827  155400  -0.184
LEV  20110827  160000  -0.308
LEV  20110827  160600  -0.281
LEV  20110827  161200  -0.275
LEV  20110827  161800  -0.272
LEV  20110827  162400  -0.361
LEV  20110827  163000  -0.385
LEV  20110827  163600  -0.272
LEV  20110827  164200  -0.284
LEV  20110827  164800  -0.298
LEV  20110827  165400  -0.363
LEV  20110827  170000  -0.403
LEV  20110827  170600  -0.324
LEV  20110827  171200  -0.371
LEV  20110827  171800  -0.274
LEV  20110827  172400  -0.295
LEV  20110827  173000  -0.266
LEV  20110827  173600  -0.195
LEV  20110827  174200  -0.228
LEV  20110827  174800  -0.191
LEV  20110827  175400  -0.191
LEV  20110827  180000  -0.230
LEV  20110827  180600  -0.163
LEV  20110827  181200  -0.182
LEV  20110827  181800  -0.077
LEV  20110827  182400  -0.167
LEV  20110827  183000  -0.095
LEV  20110827  183600  -0.035
LEV  20110827  184200  0.024
LEV  20110827  184800  0.049
LEV  20110827  185400  0.072
LEV  20110827  190000  0.038
LEV  20110827  190600  0.033
LEV  20110827  191200  0.117
LEV  20110827  191800  0.124
LEV  20110827  192400  0.182
LEV  20110827  193000  0.246
LEV  20110827  193600  0.214
LEV  20110827  194200  0.219
LEV  20110827  194800  0.309
LEV  20110827  195400  0.281
LEV  20110827  200000  0.280
LEV  20110827  200600  0.323
LEV  20110827  201200  0.400
LEV  20110827  201800  0.376
LEV  20110827  202400  0.483
LEV  20110827  203000  0.421
LEV  20110827  203600  0.493
LEV  20110827  204200  0.502
LEV  20110827  204800  0.559
LEV  20110827  205400  0.545
LEV  20110827  210000  0.625
LEV  20110827  210600  0.560
LEV  20110827  211200  0.529
LEV  20110827  211800  0.542
LEV  20110827  212400  0.554
LEV  20110827  213000  0.558
LEV  20110827  213600  0.565
LEV  20110827  214200  0.493
LEV  20110827  214800  0.491
LEV  20110827  215400  0.511
LEV  20110827  220000  0.532
LEV  20110827  220600  0.483
LEV  20110827  221200  0.500
LEV  20110827  221800  0.493
LEV  20110827  222400  0.522
LEV  20110827  223000  0.468
LEV  20110827  223600  0.475
LEV  20110827  224200  0.443
LEV  20110827  224800  0.428
LEV  20110827  225400  0.385
LEV  20110827  230000  0.400
LEV  20110827  230600  0.328
LEV  20110827  231200  0.378
LEV  20110827  231800  0.262
LEV  20110827  232400  0.299
LEV  20110827  233000  0.256
LEV  20110827  233600  0.261
LEV  20110827  234200  0.270
LEV  20110827  234800  0.261
LEV  20110827  235400  0.218
LEV  20110828  000000  0.214
LEV  20110828  000600  0.194
LEV  20110828  001200  0.133
LEV  20110828  001800  0.086
LEV  20110828  002400  0.087
LEV  20110828  003000  0.066
LEV  20110828  003600  0.046
LEV  20110828  004200  -0.029
LEV  20110828  004800  -0.024
LEV  20110828  005400  -0.011
LEV  20110828  010000  -0.022
LEV  20110828  010600  -0.023
LEV  20110828  011200  -0.027
LEV  20110828  011800  -0.061
LEV  20110828  012400  -0.101
LEV  20110828  013000  -0.115
LEV  20110828  013600  -0.141
LEV  20110828  014200  -0.238
LEV  20110828  014800  -0.278
LEV  20110828  015400  -0.287
LEV  20110828  020000  -0.323
LEV  20110828  020600  -0.392
LEV  20110828  021200  -0.391
LEV  20110828  021800  -0.428
LEV  20110828  022400  -0.500
LEV  20110828  023000  -0.558
LEV  20110828  023600  -0.573
LEV  20110828  024200  -0.551
LEV  20110828  024800  -0.607
LEV  20110828  025400  -0.610
LEV  20110828  030000  -0.703
LEV  20110828  030600  -0.742
LEV  20110828  031200  -0.753
LEV  20110828  031800  -0.777
LEV  20110828  032400  -0.811
LEV  20110828  033000  -0.896
LEV  20110828  033600  -0.958
LEV  20110828  034200  -0.991
LEV  20110828  034800  -1.034
LEV  20110828  035400  -1.049
LEV  20110828  040000  -1.081
LEV  20110828  040600  -1.091
LEV  20110828  041200  -1.090
LEV  20110828  041800  -1.096
LEV  20110828  042400  -1.101
LEV  20110828  043000  -1.159
LEV  20110828  043600  -1.141
LEV  20110828  044200  -1.151
LEV  20110828  044800  -1.148
LEV  20110828  045400  -1.149
LEV  20110828  050000  -1.166
LEV  20110828  050600  -1.194
LEV  20110828  051200  -1.175
LEV  20110828  051800  -1.158
LEV  20110828  052400  -1.192
LEV  20110828  053000  -1.181
LEV  20110828  053600  -1.181
LEV  20110828  054200  -1.160
LEV  20110828  054800  -1.139
LEV  20110828  055400  -1.143
LEV  20110828  060000  -1.090
LEV  20110828  060600  -1.047
LEV  20110828  061200  -1.053
LEV  20110828  061800  -1.055
LEV  20110828  062400  -1.033
LEV  20110828  063000  -1.013
LEV  20110828  063600  -1.015
LEV  20110828  064200  -1.005
LEV  20110828  064800  -0.948
LEV  20110828  065400  -0.945
LEV  20110828  070000  -0.987
LEV  20110828  070600  -0.948
LEV  20110828  071200  -0.906
LEV  20110828  071800  -0.868
LEV  20110828  072400  -0.839
LEV  20110828  073000  -0.813
LEV  20110828  073600  -0.748
LEV  20110828  074200  -0.729
LEV  20110828  074800  -0.677
LEV  20110828  075400  -0.661
LEV  20110828  080000  -0.614
LEV  20110828  080600  -0.606
LEV  20110828  081200  -0.594
LEV  20110828  081800  -0.557
LEV  20110828  082400  -0.511
LEV  20110828  083000  -0.452
LEV  20110828  083600  -0.426
LEV  20110828  084200  -0.375
LEV  20110828  084800  -0.337
LEV  20110828  085400  -0.302
LEV  20110828  090000  -0.285
LEV  20110828  090600  -0.241
LEV  20110828  091200  -0.195
LEV  20110828  091800  -0.178
LEV  20110828  092400  -0.152
LEV  20110828  093000  -0.129
LEV  20110828  093600  -0.119
LEV  20110828  094200  -0.065
LEV  20110828  094800  -0.041
LEV  20110828  095400  0.001
LEV  20110828  100000  0.019
LEV  20110828  100600  0.018
LEV  20110828  101200  0.050
LEV  20110828  101800  0.083
LEV  20110828  102400  0.082
LEV  20110828  103000  0.125
LEV  20110828  103600  0.120
LEV  20110828  104200  0.137
LEV  20110828  104800  0.139
LEV  20110828  105400  0.155
LEV  20110828  110000  0.143
LEV  20110828  110600  0.129
LEV  20110828  111200  0.164
LEV  20110828  111800  0.163
LEV  20110828  112400  0.164
LEV  20110828  113000  0.141
LEV  20110828  113600  0.138
LEV  20110828  114200  0.124
LEV  20110828  114800  0.123
LEV  20110828  115400  0.115
LEV  20110828  120000  0.109
LEV  20110828  120600  0.091
LEV  20110828  121200  0.059
LEV  20110828  121800  0.036
LEV  20110828  122400  0.007
LEV  20110828  123000  0.002
LEV  20110828  123600  0.008
LEV  20110828  124200  0.002
LEV  20110828  124800  -0.008
LEV  20110828  125400  -0.054
LEV  20110828  130000  -0.085
LEV  20110828  130600  -0.106
LEV  20110828  131200  -0.139
LEV  20110828  131800  -0.155
LEV  20110828  132400  -0.179
LEV  20110828  133000  -0.215
LEV  20110828  133600  -0.223
LEV  20110828  134200  -0.244
LEV  20110828  134800  -0.272
LEV  20110828  135400  -0.307
LEV  20110828  140000  -0.354
LEV  20110828  140600  -0.380
LEV  20110828  141200  -0.403
LEV  20110828  141800  -0.426
LEV  20110828  142400  -0.476
LEV  20110828  143000  -0.468
LEV  20110828  143600  -0.474
LEV  20110828  144200  -0.506
LEV  20110828  144800  -0.506
LEV  20110828  145400  -0.539
LEV  20110828  150000  -0.557
LEV  20110828  150600  -0.575
LEV  20110828  151200  -0.601
LEV  20110828  151800  -0.664
LEV  20110828  152400  -0.696
LEV  20110828  153000  -0.723
LEV  20110828  153600  -0.741
LEV  20110828  154200  -0.766
LEV  20110828  154800  -0.800
LEV  20110828  155400  -0.834
LEV  20110828  160000  -0.867
LEV  20110828  160600  -0.884
LEV  20110828  161200  -0.886
LEV  20110828  161800  -0.911
LEV  20110828  162400  -0.950
LEV  20110828  163000  -0.975
LEV  20110828  163600  -0.986
LEV  20110828  164200  -0.984
LEV  20110828  164800  -0.969
LEV  20110828  165400  -0.987
LEV  20110828  170000  -0.996
LEV  20110828  170600  -0.984
LEV  20110828  171200  -0.997
LEV  20110828  171800  -0.998
LEV  20110828  172400  -1.003
LEV  20110828  173000  -1.003
LEV  20110828  173600  -1.009
LEV  20110828  174200  -0.989
LEV  20110828  174800  -0.969
LEV  20110828  175400  -0.964
LEV  20110828  180000  -0.956
LEV  20110828  180600  -0.942
LEV  20110828  181200  -0.930
LEV  20110828  181800  -0.942
LEV  20110828  182400  -0.924
LEV  20110828  183000  -0.891
LEV  20110828  183600  -0.873
LEV  20110828  184200  -0.878
LEV  20110828  184800  -0.857
LEV  20110828  185400  -0.836
LEV  20110828  190000  -0.804
LEV  20110828  190600  -0.772
LEV  20110828  191200  -0.726
LEV  20110828  191800  -0.721
LEV  20110828  192400  -0.696
LEV  20110828  193000  -0.679
LEV  20110828  193600  -0.659
LEV  20110828  194200  -0.610
LEV  20110828  194800  -0.567
LEV  20110828  195400  -0.538
LEV  20110828  200000  -0.519
LEV  20110828  200600  -0.494
LEV  20110828  201200  -0.469
LEV  20110828  201800  -0.425
LEV  20110828  202400  -0.409
LEV  20110828  203000  -0.384
LEV  20110828  203600  -0.344
LEV  20110828  204200  -0.309
LEV  20110828  204800  -0.277
LEV  20110828  205400  -0.242
LEV  20110828  210000  -0.231
LEV  20110828  210600  -0.190
LEV  20110828  211200  -0.161
LEV  20110828  211800  -0.133
LEV  20110828  212400  -0.102
LEV  20110828  213000  -0.073
LEV  20110828  213600  -0.038
LEV  20110828  214200  0.012
LEV  20110828  214800  0.038
LEV  20110828  215400  0.069
LEV  20110828  220000  0.092
LEV  20110828  220600  0.127
LEV  20110828  221200  0.135
LEV  20110828  221800  0.154
LEV  20110828  222400  0.174
LEV  20110828  223000  0.210
LEV  20110828  223600  0.227
LEV  20110828  224200  0.247
LEV  20110828  224800  0.260
LEV  20110828  225400  0.264
LEV  20110828  230000  0.252
LEV  20110828  230600  0.258
LEV  20110828  231200  0.253
LEV  20110828  231800  0.261
LEV  20110828  232400  0.266
LEV  20110828  233000  0.266
LEV  20110828  233600  0.262
LEV  20110828  234200  0.259
LEV  20110828  234800  0.243
LEV  20110828  235400  0.240
$   'CUR' 19680606 073125    2.0    25.
   'WND' 20110826 000000  7.1     8.0    0.0
   'WND' 20110826 003000  7.5     7.0    0.0
   'WND' 20110826 010000  6.6     9.0    0.0
   'WND' 20110826 013000  5.4    13.0    0.0
   'WND' 20110826 020000  5.5   359.0    0.0
   'WND' 20110826 023000  4.9     1.0    0.0
   'WND' 20110826 030000  4.0   355.0    0.0
   'WND' 20110826 033000  3.8   359.0    0.0
   'WND' 20110826 040000  3.6   352.0    0.0
   'WND' 20110826 040000  4.2   332.0    0.0
   'WND' 20110826 050000  4.6   346.0    0.0
   'WND' 20110826 053000  4.8   344.0    0.0
   'WND' 20110826 060000  5.3   351.0    0.0
   'WND' 20110826 063000  3.7   355.0    0.0
   'WND' 20110826 070000  4.1   343.0    0.0
   'WND' 20110826 073000  4.3   339.0    0.0
   'WND' 20110826 080000  1.5   325.0    0.0
   'WND' 20110826 083000  2.1   322.0    0.0
   'WND' 20110826 090000  1.2   287.0    0.0
   'WND' 20110826 093000  1.6   202.0    0.0
   'WND' 20110826 100000  2.5   239.0    0.0
   'WND' 20110826 103000  2.3   280.0    0.0
   'WND' 20110826 110000  1.7   253.0    0.0
   'WND' 20110826 113000  3.1   246.0    0.0
   'WND' 20110826 120000  2.8   263.0    0.0
   'WND' 20110826 123000  3.2   234.0    0.0
   'WND' 20110826 130000  3.3   255.0    0.0
   'WND' 20110826 133000  2.8   255.0    0.0
   'WND' 20110826 140000  3.8   238.0    0.0
   'WND' 20110826 143000  4.4   235.0    0.0
   'WND' 20110826 150000  4.6   227.0    0.0
   'WND' 20110826 153000  4.9   238.0    0.0
   'WND' 20110826 160000  5.3   240.0    0.0
   'WND' 20110826 163000  5.9   262.0    0.0
   'WND' 20110826 170000  6.2   257.0    0.0
   'WND' 20110826 173000  6.4   257.0    0.0
   'WND' 20110826 180000  6.8   266.0    0.0
   'WND' 20110826 183000  6.5   267.0    0.0
   'WND' 20110826 190000  7.1   263.0    0.0
   'WND' 20110826 193000  7.8   272.0    0.0
   'WND' 20110826 200000  8.2   263.0    0.0
   'WND' 20110826 203000  7.8   262.0    0.0
   'WND' 20110826 210000  7.9   264.0    0.0
   'WND' 20110826 213000  7.7   267.0    0.0
   'WND' 20110826 220000  8.6   282.0    0.0
   'WND' 20110826 223000  8.7   284.0    0.0
   'WND' 20110826 230000  8.5   279.0    0.0
   'WND' 20110826 233000  8.9   280.0    0.0
   'WND' 20110827 000000  9.0   272.0    0.0
   'WND' 20110827 003000  9.2   265.0    0.0
   'WND' 20110827 010000  9.8   261.0    0.0
   'WND' 20110827 013000 10.6   245.0    0.0
   'WND' 20110827 020000 11.6   238.0    0.0
   'WND' 20110827 023000 12.4   242.0    0.0
   'WND' 20110827 030000 13.5   251.0    0.0
   'WND' 20110827 033000 13.1   238.0    0.0
   'WND' 20110827 040000 11.8   238.0    0.0
   'WND' 20110827 040000 12.9   244.0    0.0
   'WND' 20110827 050000 14.7   245.0    0.0
   'WND' 20110827 053000 14.6   252.0    0.0
   'WND' 20110827 060000 14.6   253.0    0.0
   'WND' 20110827 063000 14.9   263.0    0.0
   'WND' 20110827 070000 13.1   260.0    0.0
   'WND' 20110827 073000 15.8   260.0    0.0
   'WND' 20110827 080000 18.7   275.0    0.0
   'WND' 20110827 083000 13.7   264.0    0.0
   'WND' 20110827 090000 15.3   282.0    0.0
   'WND' 20110827 093000 15.7   261.0    0.0
   'WND' 20110827 100000 16.3   256.0    0.0
   'WND' 20110827 103000 16.6   257.0    0.0
   'WND' 20110827 110000 17.0   264.0    0.0
   'WND' 20110827 113000 19.2   256.0    0.0
   'WND' 20110827 120000 19.9   255.0    0.0
   'WND' 20110827 123000 21.0   259.0    0.0
   'WND' 20110827 130000 20.8   261.0    0.0
   'WND' 20110827 133000 22.2   269.0    0.0
   'WND' 20110827 140000 21.0   274.0    0.0
   'WND' 20110827 143000 21.9   272.0    0.0
   'WND' 20110827 150000 19.8   278.0    0.0
   'WND' 20110827 153000 21.7   276.0    0.0
   'WND' 20110827 160000 22.0   280.0    0.0
   'WND' 20110827 163000 20.9   286.0    0.0
   'WND' 20110827 170000 20.1   290.0    0.0
   'WND' 20110827 173000 21.4   295.0    0.0
   'WND' 20110827 180000 21.3   297.0    0.0
   'WND' 20110827 183000 22.6   294.0    0.0
   'WND' 20110827 190000 22.7   294.0    0.0
   'WND' 20110827 193000 24.6   297.0    0.0
   'WND' 20110827 200000 26.3   306.0    0.0
   'WND' 20110827 203000 26.2   324.0    0.0
   'WND' 20110827 210000 28.9   345.0    0.0
   'WND' 20110827 213000 23.9    18.0    0.0
   'WND' 20110827 220000 20.4    39.0    0.0
   'WND' 20110827 223000 20.2    51.0    0.0
   'WND' 20110827 230000 21.3    59.0    0.0
   'WND' 20110827 233000 21.0    61.0    0.0
   'WND' 20110828 000000 20.2    69.0    0.0
   'WND' 20110828 003000 17.9    76.0    0.0
   'WND' 20110828 010000 18.4    82.0    0.0
   'WND' 20110828 013000 17.5    86.0    0.0
   'WND' 20110828 020000 17.5    86.0    0.0
   'WND' 20110828 023000 16.5    83.0    0.0
   'WND' 20110828 030000 17.2    84.0    0.0
   'WND' 20110828 033000 18.1    79.0    0.0
   'WND' 20110828 040000 18.4    75.0    0.0
   'WND' 20110828 040000 20.6    74.0    0.0
   'WND' 20110828 050000 20.1    78.0    0.0
   'WND' 20110828 053000 19.5    76.0    0.0
   'WND' 20110828 060000 17.4    78.0    0.0
   'WND' 20110828 063000 19.4    77.0    0.0
   'WND' 20110828 070000 17.0    76.0    0.0
   'WND' 20110828 073000 15.5    77.0    0.0
   'WND' 20110828 080000 14.8    76.0    0.0
   'WND' 20110828 083000 14.5    73.0    0.0
   'WND' 20110828 090000 13.2    72.0    0.0
   'WND' 20110828 093000 13.3    70.0    0.0
   'WND' 20110828 100000 12.6    74.0    0.0
   'WND' 20110828 103000 10.8    69.0    0.0
   'WND' 20110828 110000  9.8    66.0    0.0
   'WND' 20110828 113000  8.6    67.0    0.0
   'WND' 20110828 120000  9.0    61.0    0.0
   'WND' 20110828 123000  9.4    59.0    0.0
   'WND' 20110828 130000  9.9    58.0    0.0
   'WND' 20110828 133000  8.4    67.0    0.0
   'WND' 20110828 140000  6.8    72.0    0.0
   'WND' 20110828 143000  6.0    75.0    0.0
   'WND' 20110828 150000  6.0    79.0    0.0
   'WND' 20110828 153000  6.5    68.0    0.0
   'WND' 20110828 160000  5.5    71.0    0.0
   'WND' 20110828 163000  4.6    75.0    0.0
   'WND' 20110828 170000  4.6    66.0    0.0
   'WND' 20110828 173000  4.9    56.0    0.0
   'WND' 20110828 180000  4.3    61.0    0.0
   'WND' 20110828 183000  4.0    53.0    0.0
   'WND' 20110828 190000  3.1    43.0    0.0
   'WND' 20110828 193000  2.3    52.0    0.0
   'WND' 20110828 200000  2.7    44.0    0.0
   'WND' 20110828 203000  4.7   332.0    0.0
   'WND' 20110828 210000  4.9   324.0    0.0
   'WND' 20110828 213000  4.7   338.0    0.0
   'WND' 20110828 220000  4.7   336.0    0.0
   'WND' 20110828 223000  4.3   334.0    0.0
   'WND' 20110828 230000  4.6   335.0    0.0
   'WND' 20110828 233000  4.6   347.0    0.0
   'WND' 20110829 000000  4.9   347.0    0.0
   'STP'
$
$ -------------------------------------------------------------------- $
$ End of input file                                                    $
$ -------------------------------------------------------------------- $