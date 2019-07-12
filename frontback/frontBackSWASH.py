"""
This script holds the master function for the simulation Setup
for the Swash model setup
"""
import testbedutils.anglesLib
from prepdata import inputOutput
from prepdata.prepDataLib import PrepDataTools as STPD
from getdatatestbed.getDataFRF import getDataTestBed
from getdatatestbed.getDataFRF import getObs
import datetime as DT
import os, glob, makenc, pickle, tarfile
import netCDF4 as nc
import numpy as np
from prepdata import prepDataLib as STPD
from prepdata.inputOutput import swashIO
from getdatatestbed import getDataFRF
import plotting.operationalPlots as oP
from testbedutils import sblib as sb
from testbedutils import waveLib as sbwave
from plotting.operationalPlots import obs_V_mod_TS
from testbedutils import geoprocess as gp
#import multiprocessing
#import matplotlib
#matplotlib.use('Agg')


def SwashSimSetup(startTime, inputDict):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the Swash wave/FLow model


    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function

    """
    # begin by setting up input parameters
    timerun = inputDict.get('simulationDuration', 24)
    pFlag = inputDict.get('pFlag', True)
    server = inputDict.get('THREDDS', 'CHL')
    # this raises error if not present (intended)
    version_prefix = inputDict['version_prefix']
    path_prefix = inputDict['path_prefix']  # data super directiory
    # ______________________________________________________________________________
    # define version parameters
    versionlist = ['base', 'ts']
    assert version_prefix.lower() in versionlist, 'Please check your version Prefix'
    # here is where we set something that would handle 3D mode or time series mode,
    # might set flags for preprocessing below
    # _______________________________________________________________________________
    # set times
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    date_str = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime

    if isinstance(timerun, str):
        timerun = int(timerun)

    # __________________Make Working Data Directories_____________________________________________
    #
    if not os.path.exists(os.path.join(path_prefix, date_str)):  # if it doesn't exist
        os.makedirs(os.path.join(path_prefix, date_str))  # make the directory
    if not os.path.exists(os.path.join(path_prefix, date_str)):
        os.makedirs(os.path.join(path_prefix, date_str))

    print("Model Time Start : %s  Model Time End:  %s" % (d1, d2))
    print("OPERATIONAL files will be place in {} folder".format(os.path.join(path_prefix, date_str)))
    # ______________________________________________________________________________
    # begin model data gathering
    go = getObs(d1, d2, THREDDS=server)                  # initialize get observation class
    prepdata = STPD.PrepDataTools()                      # for preprocessing
    gdTB = getDataTestBed(d1, d2, THREDDS=server)        # for bathy data gathering
    # _____________WAVES____________________________
    print('_________________\nGetting Wave Data')
    rawspec = go.getWaveSpec(gaugenumber= '8m-array') #'AWAC-11m')  #
    assert 'time' in rawspec, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (d1, d2)
    # preprocess wave spectra
    if version_prefix.lower() == 'base':
        wavepacket = prepdata.prep_SWASH_spec(rawspec, version_prefix)
    else:
        raise NotImplementedError('pre-process TS data ')
    # _____________WINDS______________________
    print('_________________\nSkipping Wind')
    # try:
    #     rawwind = go.getWind(gaugenumber=0)
    #     # average and rotate winds
    #     windpacket = prepdata.prep_wind(rawwind, wavepacket['epochtime'])
    #     # wind height correction
    #     print('number of wind records %d with %d interpolated points' % (
    #         np.size(windpacket['time']), sum(windpacket['flag'])))
    # except (RuntimeError, TypeError):
    #     windpacket = None
    #     print(' NO WIND ON RECORD')

    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    try:
        # get water level data
        rawWL = go.getWL()
        # average WL
        WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])
        print('number of WL records %d, with %d interpolated points' % (
            np.size(WLpacket['time']), sum(WLpacket['flag'])))
    except (RuntimeError, TypeError):
        WLpacket = None
    ### ____________ Get bathy grid from thredds ________________
    # bathy = gdTB.getGridSwash(method='historical')
    bathy = gdTB.getBathyIntegratedTransect(method=1, ybound=[940, 950])
    swsinfo, gridDict = prepdata.prep_SwashBathy(wavepacket['xFRF'], wavepacket['yFRF'], bathy, dx=1, dy=1,
                                                 yBounds=[944, 947])  # non-inclusive index if you want 3 make 4 wide

    ## begin output
    # set some of the class instance variables before writing Sws file
    swio = swashIO(WL=WLpacket['avgWL'], equilbTime=wavepacket['spinUp'],
                   Hs=wavepacket['Hs'], Tp=1/wavepacket['peakf'],
                   Dm=wavepacket['waveDm'], fileNameBase=date_str,
                   path_prefix=path_prefix, version_prefix=version_prefix,
                   nprocess=gridDict['h'].shape[0])

    # write SWS file first
    swio.write_sws(swsinfo)
    swio.write_spec1D(wavepacket['freqbins'], wavepacket['fspec'])
    swio.write_bot(gridDict['h'])
    # now write QA/
    print(' TODO: figure OUt how to write flags, and what flags to write!!')
    swio.flags = None
    pickleName = os.path.join(path_prefix, date_str,'.pickle')
    with open(pickleName, 'wb') as fid:
        pickle.dump(swio, fid, protocol=pickle.HIGHEST_PROTOCOL)
    return swio

def SwashAnalyze(startTime, inputDict, swio):
    """This runs the post process script for Swash wave will create plots and netcdf files at request

    Args:
        inputDict (dict): this is an input dictionary that was generated with the
            keys from the project input yaml file
        startTime (str): input start time with datestring in format YYYY-mm-ddThh:mm:ssZ


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    print("check docstrings for Analyze and preprocess")
    # ___________________define Global Variables___________________________________

    pFlag = inputDict.get('pFlag', True)
    version_prefix = inputDict['version_prefix']
    Thredds_Base = inputDict.get('netCDFdir', '/thredds_data')
    server = inputDict.get('THREDDS', 'CHL')
    # the below should error if not included in input Dict
    path_prefix = inputDict['path_prefix']  # for organizing data
    simulationDuration = inputDict['simulationDuration']
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, datestring)
    model = 'Swash'
    # ____________________________________________________________________________
    #________________________________________________________

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    go = getDataFRF.getObs(d1, d2, server)  # setting up get data instance
    prepdata = STPD.PrepDataTools()  # initializing instance for rotation scheme
    swio = swio                             # initializing read/write class as passed (has previous info from setup)

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################

    matfile = os.path.join(swio.path_prefix, ''.join(swio.ofileNameBase.split('-'))+'.mat')
    print(' TODO: write run wall time to output file')
    print('Loading files ')
    simData, simMeta = swio.loadSwash_Mat(fname=matfile)  # load all files

    ######################################################################################################################
    #################################   obtain total water level   #######################################################
    ######################################################################################################################

    eta = simData['eta'].squeeze()

    # now adapting Chuan's runup code, here we use 0.08 m for runup threshold
    r_depth = 0.08  # 4.0 * np.nanmax(np.abs(h[runupInd][1:] - h[runupInd][:-1]))

    # Preallocate runup variable
    runup = np.zeros(eta.shape[0])
    x_runup = np.zeros_like(runup)

    for aa in range(runup.shape[0]):
        # Water depth
        wdepth = eta[aa, :] + simData['elevation']
        # Find the runup contour (search from left to right)
        wdepth_ind = np.argmin(abs(wdepth - r_depth))  # changed from Chuan's original code
        # Store the water surface elevation in matrix
        runup[aa] = eta[aa, wdepth_ind]  # unrealistic values for large r_depth
        # runup[aa]= -h[wdepth_ind]
        # Store runup position
        x_runup[aa] = simData['xFRF'][wdepth_ind]
    maxRunup = np.amax(runup)

    ######################################################################################################################
    ######################################################################################################################
    ##################################  plotting #########################################################################
    ######################################################################################################################
    ######################################################################################################################
    nSubSample = 10
    if not os.path.exists(os.path.join(path_prefix,datestring, 'figures')):
        os.makedirs(os.path.join(path_prefix,datestring, 'figures'))  # make the directory for the simulation plots
    figureBaseFname = 'CMTB_waveModels_{}_{}_'.format(model, version_prefix)
    from matplotlib import pyplot as plt
    SeaSwellCutoff = 0.05
    # if pFlag == True:
    #     ## remove images before making them
    #     imgList = glob.glob(os.path.join(path_prefix, datestring, 'figures', '*.png'))
    #     [os.remove(ff) for ff in imgList]
    #
    #     ############### write a parallel data load function ##################
    #     dataOut = []
    #
    #     def parallel_generateCrossShoreTimeSeries(tidx):
    #         ## generate a function that operates with only one input, can access local variable space
    #         timeStep = simData['time'][tidx]
    #         ofPlotName = os.path.join(path_prefix, datestring, 'figures',
    #                                   figureBaseFname + 'TS_' + timeStep.strftime('%Y%m%dT%H%M%S%fZ') + '.png')
    #         oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'],
    #                                          simData['xFRF'])
    #         dataOut.append(ofPlotName)
    #
    #     ######################################################################
    #     tstart = DT.datetime.now()
    #     pool = multiprocessing.Pool(4)  # open multiprocessing pool
    #     dataBack = pool.map(parallel_generateCrossShoreTimeSeries, range(0, len(simData['time']), nSubSample))
    #     pool.close()
    #     print('Took {} long to make all the plots in parallel {} processors'.format(DT.datetime.now() - tstart, 4))
    #
    #     # ## now make gif of waves moving across shore*
    #     imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '*TS_*.png')))
    #     sb.makegif(imgList,
    #                os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.gif'.format(datestring)),
    #                dt=0.1)
    #     print('Took {} long to make the movie and all the plots '.format(DT.datetime.now() - tstart))
    #     # finish making images by taring files
    #     tarOutFile = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS.tar.gz')
    #     sb.myTarMaker(tarOutFile, imgList)
    #     print('Took {} long to make all the plots, movie and tarball '.format(DT.datetime.now() - tstart))
    # make function for processing timeseries data
    fspec, freqs = sbwave.timeSeriesAnalysis1D(simData['time'].squeeze(), simData['eta'].squeeze(), bandAvg=6)
    total = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=None)
    SeaSwellStats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
    IGstats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)
    HsTS = 4 * np.std(simData['eta'].squeeze(), axis=0)

    #############################################################################################################
    ####################################### Plot functions ######################################################
    #############################################################################################################
    setup = np.mean(simData['eta'], axis=0).squeeze()
    WL = simMeta['WL'] #added in editing, should possibly be changed?
    if pFlag == True:
        from plotting import operationalPlots as oP
        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TempFname.png')
        oP.plotCrossShoreSummaryTS(ofname, simData['xFRF'], simData['elevation'], total,
                               SeaSwellStats, IGstats, setup=setup, WL=WL)
        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_spectrograph.png')
        oP.crossShoreSpectrograph(ofname, simData['xFRF'], freqs, fspec)
        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_surfaceTimeseries.png')
        oP.crossShoreSurfaceTS2D(ofname, simData['eta'], simData['xFRF'], simData['time'])
        # plt.figure()
        # plt.plot()
        #
        # plot time-series of cross-shore evolution movie
        print('this script makes double pictures, we can probably sub-sample')

        for tidx, timeStep in enumerate(simData['time']):
            ofPlotName = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_' + timeStep.strftime('%Y%m%dT%H%M%S%fZ') +'.png')
            oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'], simData['xFRF'])
        # now make gif of waves moving across shore
        imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', '*_TS_*.png')))
        sb.makegif(imgList, os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.gif'.format(datestring)),dt=0.1)

        tarOutFile = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS.tar.gz')
        sb.myTarMaker(tarOutFile, imgList)

        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TimeStack.png')
        ## figure making
        cmap = 'RdBu'
        plt.figure()
        mappable = plt.pcolor(simData['xFRF'], simData['time'], simData['eta'].squeeze(), cmap=cmap)
        plt.colorbar(mappable)
        plt.title('Time Stack for {} transect'.format(simData['yFRF']))
        plt.ylabel('cross-shore position')
        plt.xlabel('simulation time(s)')
        plt.savefig(ofname)
        plt.close()

    # ################################
    #        Make NETCDF files       #
    # ################################
    ## before netCDF.
    # get significant wave height for cross shore
    # slice the time series so we're on ly isolating the non-repeating time series of data
    #################################################3
    if len(simData['time']) == 4097:
        tsTime = np.arange(0, 2048.5, 0.5)
    else:
        tsTime = np.arange(0, 2048, 1)
    fldrArch = os.path.join(model, version_prefix)
    spatial = {'time': nc.date2num(d1, units='seconds since 1970-01-01 00:00:00'),
               'station_name': 'Regional Simulation Field Data',
               'tsTime': tsTime,
               'waveHsIG': np.reshape(IGstats['Hm0'], (1, len(simData['xFRF']))),
               'eta': np.swapaxes(simData['eta'], 0, 1),
               'totalWaterLevel': maxRunup,
               'totalWaterLevelTS': np.reshape(runup, (1, len(runup))),##
               'velocityU': np.swapaxes(simData['velocityU'], 0, 1),
               'velocityV': np.swapaxes(simData['velocityV'], 0, 1),
               'waveHs': np.reshape(SeaSwellStats['Hm0'], (1, len(simData['xFRF']))),#or from HsTS??
               'xFRF': simData['xFRF'],
               'yFRF': simData['yFRF'][0],
               'DX': int(simData['xFRF'][1]-simData['xFRF'][0]),
               'DY': 1,#must be adjusted for 2D simulations
               'NI': len(simData['xFRF']),
               'NJ': 1,#must be adjusted for 2D simulations
               }
    #'eta': np.transpose(simData['eta'], (len(simData['time']), len(tsStartSim), len(simData['xFRF'])))
    TdsFldrBase = os.path.join(Thredds_Base, fldrArch)
    NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Field'), datestring, model=model)
    # make the name of this nc file
    NCname = 'CMTB-waveModels_{}_{}_Field_{}.nc'.format(model, version_prefix, datestring)
    fieldOfname = os.path.join(NCpath,
                               NCname)  # TdsFldrBase + '/CMTB-waveModels_Swash_{}_Local-Field_%s.nc'.format(version_prefix, datestring)

    if not os.path.exists(TdsFldrBase):
        os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    if not os.path.exists(os.path.join(TdsFldrBase, 'Field', 'Field.ncml')):
        inputOutput.makencml(os.path.join(TdsFldrBase, 'Field', 'Field.ncml'))  # remake the ncml if its not there
    # make file name strings
    flagfname = os.path.join(fpath, 'Flags{}.out.txt'.format(datestring))  # startTime # the name of flag file
    fieldYaml = 'yaml_files/waveModels/%s/swash_global.yml' % (model)  # field
    varYaml = 'yaml_files/waveModels/%s/swash_var.yml' % (model)
    assert os.path.isfile(fieldYaml), 'NetCDF yaml files are not created'  # make sure yaml file is in place
    makenc.makenc_phaseresolved(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)

