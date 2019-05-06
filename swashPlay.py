import matplotlib
from testbedutils import waveLib as wl
# matplotlib.use('Agg')
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
swio = inputOutput.SWASHio(datestring, WL = WL)
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
################################## Make Plot functions ######################################################
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


def myDataloadfunction(data):
    """ Load SWASH output from

    Args:
        data:

    Returns:

    """
    dataOUT, dataTemp, runMeta = {}, {}, {}
    # initalize outDict with empty arrays
    times, nonTimeDependant = [], []
    # first load all vars and parse as time dependent array
    for dictInKey, dataValues in data.items():  ## should i sort these?
        origVarName = dictInKey.split('_')
        saveKey = ''.join(origVarName[:-2])
        # print('Looking through var {} will save as {} or {}'.format(origVarName, saveKey, dictInKey))
        try:
            timeString = origVarName[-2] + '_' + origVarName[-1]
            try:
                d = DT.datetime.strptime(timeString, "%H%M%S_%f")
                times.append(DT.timedelta(hours=d.hour, minutes=d.minute, seconds=d.second, microseconds=d.microsecond))
                # print('saved as time step {}'.format(timeString))
            except ValueError:  # this will be things like __header__, __global__... etc, erros out on d = ...
                runMeta[origVarName[2]] = dataValues
                continue

            ######## now write data to temp dictionary
            # save data as 1D arrays, to reshape later
            # print('Original key {} put {} values into key {}'.format(origVarName, dataValues.shape, saveKey))
            if saveKey not in dataTemp:  # then haven't parsed it before, just add it
                dataTemp[saveKey] = [dataValues]
            else:  # just add the data !
                dataTemp[saveKey].append(dataValues)

        except IndexError:  # for not time dependant variables
            # print('Original key {} put {} values into key {}'.format(origVarName, dataValues.shape, dictInKey))
            # just save them as is  (variables like Xp, because save key turns into gibberish)
            nonTimeDependant.append(dictInKey)
            dataTemp[dictInKey] = [dataValues]
            # pass

    ################### reshape list structure to 3 d array
    # parsed quickly to a single dictionary, now reshape to [t, y, x]
    # assume that we're always outputing eta and use this to get number of timesteps
    # turn nan's to masks for output
    tSteps, nY, nX = np.shape(dataTemp['Watlev'])
    t = DT.datetime.now()
    # for var in dataTemp.keys():
    #     if var not in nonTimeDependant:
    #         dataOUT[var] = np.reshape(dataTemp[saveKey], (tSteps, nY, nX))
    # print('reshape took {}'.format(DT.datetime.now() - t))
    t = DT.datetime.now()
    for var in dataTemp.keys():
        if var not in nonTimeDependant:
            dataOUT[var] = np.ma.MaskedArray(dataTemp[var], mask=np.isnan(dataTemp[var]))
    print('masked array took {}'.format(DT.datetime.now() - t))
    ##############################################################################
    # now restructure dictionary with a bunch of keys to a single
    # dictionary with multi dimensional output (concatenate the layered values)
    ##############################################################################
    for var in list(dataOUT.keys()):
        # if var in meshVars: # then variables have same number as layers
        saveKey = var.translate(var.maketrans('', '', '0123456789'))
        if saveKey != var:  # then there's not associated with it
            if saveKey in dataOUT:
                dataOUT[saveKey] = np.append(dataOUT[saveKey], np.expand_dims(dataOUT[var], axis=-1), axis=-1)
            else:
                dataOUT[saveKey] = np.expand_dims(dataOUT[var], axis=-1)
            del dataOUT[var]

    # d1 = DT.datetime.strptime(self.ofileNameBase, '%Y-%m-%dT%H%M%SZ')
    # wrap up into output dictionary named data
    dataDict = {'time': np.unique(times),  # convert each time step to date time from time delta object
                'elevation': data['Botlev'].squeeze(),
                'xFRF': data['Xp'].squeeze(),
                'yFRF': data['Yp'].squeeze(),
                'eta': dataOUT['Watlev'],
                'Ustar': dataOUT['Ustar'],
                'velocityU': dataOUT['velx'],
                'velocityUprofile': dataOUT['velkx'],
                'velocityV': dataOUT['vely'],
                'velocityVprofile': dataOUT['velky'],
                'velocityZprofile': dataOUT['w'],
                'NormPresProfile': dataOUT['Nprsk'], }
    #
    # metaDict = {'runMeta': runMeta,
    #             'waveDm': self.Dm,
    #             'waveHs': self.Hs,
    #             'waveTp': self.Tp,
    #             'WL': self.WL,
    #             'nLayers': self.nLayers,
    #             'nprocess': self.nprocess,
    #             'equlibTime': self.equlibTime,
    #             'simWallTime': DT.timedelta(seconds=self.simulationWallTime),
    #             'spreadD': self.spreadD,
    #             'versionPrefix': self.version_prefix}
    return dataDict
