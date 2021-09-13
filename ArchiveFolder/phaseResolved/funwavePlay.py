import matplotlib
# matplotlib.use('Agg')
from testbedutils import waveLib as wl
import numpy as np
import datetime as DT
from matplotlib import pyplot as plt
from prepdata import inputOutput
from plotting import operationalPlots as oP
import os, glob, multiprocessing
from testbedutils import sblib as sb
import pickle
###
nSubSample = 5
plotting = False
loadData = False  # don't load in debug mode, takes forever!
path_prefix = '/home/spike/cmtb/data/FUNWAVE'
datestring = '20151005T000000Z'
figureBaseFname = 'CMTB_waveModels_FUNWAVE_'
bathyFile = '/home/spike/repos/cmtb/data/FUNWAVE/20151005T000000Z/depth.txt'
pickleFname = os.path.join(path_prefix, figureBaseFname + 'TS.pickle')
SeaSwellCutoff = 0.04
WL = 1.4
crossShoreTransectLine = 200
bandAvg = 3
########################################3
# funwave Timeseries analysis

fwio = inputOutput.funwaveIO(datestring, WL = WL)
if loadData == True:
    flist = sorted(glob.glob(os.path.join(path_prefix, datestring,'output/sta*')))
    simData = fwio.loadFunwaveTransect(flist)
    simData['time'] = np.array([DT.datetime.strptime(datestring, '%Y%m%dT%H%M%SZ') + DT.timedelta(seconds=i) for i in simData['time']])
    simData['elevation'] = -fwio.loadBathy(bathyFile)[crossShoreTransectLine,]
    with open(pickleFname, 'wb') as fid:
        pickle.dump(simData, fid, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(pickleFname, 'rb') as fid:
        simData = pickle.load(fid)

if plotting == True:
    ## remove images before making them
    imgList = glob.glob(os.path.join(path_prefix, datestring, 'figures', '*.png'))
    [os.remove(ff) for ff in imgList]
    ############### write a parallel data load function ##################
    dataOut=[]
    def parallel_generateCrossShoreTimeSeries(tidx):
        ## generate a function that operates with only one input, can access local variable space
        timeStep = simData['time'][tidx]
        ofPlotName = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_' + timeStep.strftime('%Y%m%dT%H%M%S%fZ') +'.png')
        oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'], simData['xFRF'])
        dataOut.append(ofPlotName)
    ######################################################################
    tstart = DT.datetime.now()
    pool=multiprocessing.Pool(4)  # open multiprocessing pool
    dataBack = pool.map(parallel_generateCrossShoreTimeSeries, range(0, len(simData['time']), nSubSample))
    pool.close()
    print('Took {} long to make all the plots in parallel {} processors'.format(DT.datetime.now() - tstart, 4))

    # ## now make gif of waves moving across shore*
    imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '*TS_*.png')))
    sb.makegif(imgList, os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.gif'.format(datestring)), dt=0.1)
    print('Took {} long to make the movie and all the plots '.format(DT.datetime.now() - tstart))
    # finish making images by taring files
    tarOutFile  = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS.tar.gz')
    sb.myTarMaker(tarOutFile, imgList)
    print('Took {} long to make all the plots, movie and tarball '.format(DT.datetime.now() - tstart))

############### frequency analysis ##########################################################
fspec, freqs = wl.timeSeriesAnalysis1D(simData['time'].squeeze(),simData['eta'].squeeze(), bandAvg=bandAvg, windowLength=30)
total = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=None)
SeaSwellStats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
IGstats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)
HsTS = 4 * np.std(simData['eta'].squeeze(), axis=0)
######################################################################################################
####################################### Plot functions ######################################################
#############################################################################################################
setup = np.mean(simData['eta'], axis=0).squeeze()
from plotting import operationalPlots as oP
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_crossShoreSummary.png')
oP.plotCrossShoreSummaryTS(ofname, simData['xFRF'], simData['elevation'], total,
                           SeaSwellStats, IGstats, setup=setup, WL=WL)
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_spectrograph.png')
oP.crossShoreSpectrograph(ofname, simData['xFRF'], freqs, fspec)
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_surfaceTimeseries.png')
oP.crossShoreSurfaceTS2D(ofname, simData['eta'], simData['xFRF'], simData['time'])

