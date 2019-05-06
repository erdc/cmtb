import matplotlib
matplotlib.use('Agg')
import numpy as np
import datetime as DT
from scipy import io
from matplotlib import pyplot as plt
from prepdata import inputOutput
from plotting import operationalPlots as oP
import os, glob, multiprocessing
from testbedutils import sblib as sb
## # my two comparison files
# set global variables
datestring = '2015-10-05T000000Z'
nSubSample = 3
path_prefix = 'base'
figureBaseFname = 'Stuff'
SeaSwellCutoff = 0.05
########### initalize class from where i'm loading data from
swio = inputOutput.SWASHio(datestring, WL = 1.4)

## remove images before making them
imgList = glob.glob(os.path.join(path_prefix, datestring, 'figures', '*.png'))
[os.remove(ff) for ff in imgList]
myMat = '/home/spike/cmtb/data/SWASH/play/20151005T000000Z.mat'
simData, meta = swio.loadSwash_Mat(myMat)

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
pool=multiprocessing.Pool(4) # open multiprocessing pool
dataBack = pool.map(parallel_generateCrossShoreTimeSeries,range(0, len(simData['time']), nSubSample) )
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

from testbedutils import waveLib as wl

# make function for processing timeseries data
fspec, freqs = timeSeriesAnalysis1D(simData['eta'].squeeze(), simData['xFRF'].squeeze(), simData['time'].squeeze())
## i'm here, need to calculate setup in cross shore , maybe return from above function?
# then need to calculate cross-shore wave height (total, Sea Swell and IG)


SeaSwellStats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
IGstats = wl.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)

HsS = 4* np.std(simData['eta'].squeeze(), axis=0)

################## done function
############### make plot of surface timeseries
plt.figure();
plt.plot(simData['xFRF'], HsS, '.', label='Me')
plt.plot(simData['xFRF'], -simData['elevation'], label='bottom')
plt.legend()
fname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_Hs.png')
plt.savefig(fname)


def timeSeriesAnalysis1D(time, xFRF, eta, **kwargs):
    """process 1D timeserise analysis, function will demean data by default.  It can operate on 
    2D spatial surface elevation data, but will only do 1D analysis (not puv/2D directional waves)
    for frequency band averaging, will label with band center

    Args:
        time: time (datetime object)
        xFRF: cross-shore position in local coordindates
        eta: surface timeseries 

        **kwargs: 
            'windowLength': window length for FFT, units are minutes (Default = 10 min)
            'overlap': overlap of windows for FFT, units are percentage of window length (Default=0.75)
            'bandAvg': number of bands to average over 
            'timeAx (int): a number defining which axis in eta is time (Default = 0)

    Returns:
        fspec (array): array of power spectra, dimensioned by [space, frequency]
        frqOut (array): array of frequencys associated with fspec

    Raises: 
        Warnings if not all bands are processed (depending on size of freqeuency bands as output by FFT 
            and band averaging chosen, function will neglect last few (high frequency) bands 

    TODO:
        can add surface correction for 

    """
    from scipy.signal import welch
    import warnings
    ## kwargs below
    nPerSeg = kwargs.get('WindowLength', 10 * 60)  # window length (10 minutes in seconds)
    overlapPercentage = kwargs.get('overlap', 3 / 4)  # 75% overlap per segment
    bandAvg = kwargs.get('bandAvg', 6)  # average 6 bands
    myAx = kwargs.get('timeAx', 0)  # time dimension of eta
    overlap = overlap * overlapPercentage
    ## preprocessing steps
    etaDemeaned = np.nan_to_num(eta - np.mean(eta, axis=0))
    # etaDemeaned = np.ma.masked_array(etaD, mask=np.isnan(eta).data, fill_value=-999)   # demean surface time series
    assert eta.shape[myAx] == time.shape[0], "axis selected for eta doesn't match time"
    freqSample = np.median(np.diff(time)).total_seconds()

    freqsW, fspecW = welch(x=etaDemeaned, window='hanning', fs=freqSample, nperseg=nPerSeg, noverlap=overlap,
                           nfft=None, return_onesided=True, detrend='linear', axis=myAx)
    # remove first index of array (DC componanats)
    freqW = freqsW[1:]
    fspecW = fspecW[1:]
    ## TODO: add surface correction here

    # initalize for band averaging
    # dk = np.floor(bandAvg/2).astype(int)  # how far to look on either side of band of interest
    frqOut, fspec = [], []
    for kk in range(0, len(freqsW) - bandAvg, bandAvg):
        avgIdxs = np.linspace(kk, kk + bandAvg - 1, num=bandAvg).astype(int)
        frqOut.append(freqW[avgIdxs].sum(axis=myAx) / bandAvg)  # taking average of freq for label (band centered label)
        fspec.append(fspecW[avgIdxs].sum(axis=myAx) / bandAvg)
    if max(avgIdxs) < len(freqW):
        warnings.warn('neglected {} freq bands'.format(len(freqW) - max(avgIdxs)))

    frqOut = np.array(frqOut).T
    fspec = np.array(fspec).T
    # output as
    fspec = np.ma.masked_array(fspec, mask=np.tile((fspec == 0).all(axis=1), (frqOut.size, 1)).T)
    return fspec, frqOut

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

