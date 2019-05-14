import matplotlib
from testbedutils import waveLib as wl
matplotlib.use('Agg')
import numpy as np
import datetime as DT
from matplotlib import pyplot as plt
from prepdata import inputOutput
from plotting import operationalPlots as oP
import os, glob, multiprocessing
from testbedutils import sblib as sb
## # my two comparison files
# set global variables
datestring = '2015-10-05T000000Z'
nSubSample = 3
plotting = False
path_prefix = '/home/spike/cmtb/base'
figureBaseFname = 'Stuff_'
SeaSwellCutoff = 0.05
WL = 1.4
########### initalize class from where i'm loading data from
swio = inputOutput.swashIO(datestring, WL = WL)
myMat = '/home/spike/cmtb/data/SWASH/play/20151005T000000Z.mat'
simData, meta = swio.loadSwash_Mat(myMat)

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
# make function for processing timeseries data
fspec, freqs = wl.timeSeriesAnalysis1D(simData['time'].squeeze(),simData['eta'].squeeze(), bandAvg=6)
total = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=None)
SeaSwellStats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
IGstats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)
HsTS = 4 * np.std(simData['eta'].squeeze(), axis=0)

#############################################################################################################
####################################### Plot functions ######################################################
#############################################################################################################
setup = np.mean(simData['eta'], axis=0).squeeze()
from plotting import operationalPlots as oP
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TempFname.png')
oP.plotCrossShoreSummaryTS(ofname, simData['xFRF'], simData['elevation'], total,
                           SeaSwellStats, IGstats, setup=setup, WL=WL)
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_spectrograph.png')
oP.crossShoreSpectrograph(ofname, simData['xFRF'], freqs, fspec)
ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_surfaceTimeseries.png')
oP.crossShoreSurfaceTS2D(ofname, simData['eta'], simData['xFRF'], simData['time'])


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
