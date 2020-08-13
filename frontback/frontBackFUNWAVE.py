"""
This script holds the master function for the simulation Setup
for the Swash model setup
"""
from prepdata import inputOutput
from prepdata.prepDataLib import PrepDataTools as STPD
from getdatatestbed.getDataFRF import getDataTestBed
from getdatatestbed.getDataFRF import getObs
import datetime as DT
import os, glob, makenc, pickle, tarfile
import netCDF4 as nc
import numpy as np
from prepdata import prepDataLib as STPD
from prepdata.inputOutput import funwaveIO
import plotting.operationalPlots as oP
from testbedutils import sblib as sb
from testbedutils import waveLib as sbwave
from testbedutils import fileHandling
from plotting.operationalPlots import obs_V_mod_TS
from testbedutils import geoprocess as gp
import multiprocessing

def FunwaveSimSetup(startTime, rawWL, rawspec, bathy, inputDict):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the Swash wave/FLow model


    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function

    """
    # begin by setting up input parameters
    model = inputDict['modelSettings'].get('model')
    grid = inputDict['modelSettings'].get('grid').lower()
    timerun = inputDict.get('simulationDuration', 1)
    plotFlag = inputDict.get('plotFlag', True)
    # this raises error if not present (intended)
    version_prefix = inputDict['modelSettings']['version_prefix'].lower()
    path_prefix = inputDict['path_prefix']  # data super directory
    dx = inputDict.get('dx', 0.5)
    dy = inputDict.get('dy', 0.5)
    nf = inputDict.get('nf', 100)
    phases = inputDict.get('phases', None)
    # ______________________________________________________________________________
    # here is where we set something that would handle 3D mode or time series mode, might set flags for preprocessing below
    fileHandling.checkVersionPrefix(model=model, inputDict=inputDict)
    # _______________________________________________________________________________
    # set times
    # d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    # d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    date_str = startTime #d1.strftime('%Y-%m-%dT%H%M%SZ')
    prepdata = STPD.PrepDataTools()  # for preprocessing

    # __________________Make Working Data Directories_____________________________________________
    print("OPERATIONAL files will be place in {} folder".format(os.path.join(path_prefix, date_str)))

    # _____________WAVES____________________________
    print('_________________\nGetting Wave Data')
    assert 'time' in rawspec, "\n++++\nThere's No Wave data"
    # preprocess wave spectra


    #if version_prefix.lower() == 'base':
    #   wavepacket1 = prepdata.prep_SWASH_spec(rawspec, version_prefix, model=model, nf=inputDict['modelSettings']['nf'])

    #else:
    #    #raise NotImplementedError('pre-process TS data ')
    #    wavepacket1 = prepdata.prep_SWASH_spec(rawspec, version_prefix, model=model, nf=inputDict['modelSettings']['nf'])

    print("\n\nDEBUG GABY: line 75 of frontBackFUNWAVE repeats wavepacket when it was already used on lines 68/72 (depending on the if statement)\n\n")
    wavepacket = prepdata.prep_SWASH_spec(rawspec, version_prefix, model=model, nf=nf, phases=phases)

    # _____________WINDS______________________
    print('_________________\nSkipping Wind')
    
    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    WLpacket = prepdata.prep_WL(rawWL, rawWL['epochtime']) # time average WL

    ### ____________ Get bathy grid from thredds ________________

    if grid.lower() == '1d':    # non-inclusive index for yBounds
        ybounds = [bathy['yFRF']-1.5*dy,bathy['yFRF']+1.5*dy]# [bathy['yFRF']-dy, bathy['yFRF']+dy]  ## should take a
        # look at this
    else:
        ybounds = [600,1100]

    print("DEBUG GABY: ybounds =", ybounds)
    _, gridDict = prepdata.prep_SwashBathy(bathy['xFRF'][0], bathy['yFRF'], bathy.copy(), ybounds)  #
    # del bathy  # was carrying bathy to super function


    # _____________ begin writing files _________________________
    # set some of the class instance variables before writing input files
    # TODO: @Gaby, calculate nprocessors (px * py), i think this is based on the grid, so you can use the output from
    #  prep_FunwaveBathy

    [Nglob,Mglob] = gridDict['elevation'].shape
    px = np.floor(Mglob / 150)
    if grid.lower() == '1d':
        py = 1
    else:
        py = np.floor(Nglob / 150)
    nprocessors = px * py  # now calculated on init


    fio = funwaveIO(fileNameBase=date_str, path_prefix=path_prefix, version_prefix=version_prefix, WL=WLpacket['avgWL'],
                    equilbTime=0, Hs=wavepacket['Hs'], Tp=1/wavepacket['peakf'], Dm=wavepacket['waveDm'],
                    px=px, py=py, nprocessors=nprocessors,Mglob=Mglob,Nglob=Nglob)


    #TODO: change the below functions to write 1D simulation files with appropriate input information.  In other
    # words the dictionaries here that are input to your write functions, need to come out of the "prep" functions
    # above.  For this we'll have to modify the "prep" functions to do that.   I'm happy to help point you to where
    # you need to modify as needed.

    ## write spectra, depth, and station files
    if grid.lower() == '1d':
        fio.Write_1D_Bathy(dx, dy, gridDict['elevation'])
        fio.Write_1D_Spectra_File(wavepacket)
    else:
        fio.Write_2D_Bathy(dx, dy, gridDict['elevation'])
        fio.Write_2D_Spectra_File(wavepacket, wavepacket['amp2d'])

    ## write input file
    fio.Write_InputFile(inputDict)

    #fio.write_bot(gridDict['h'])
    # now write QA/QC flag
    fio.flags = None
    pickleName = os.path.join(path_prefix, date_str,'.pickle')
    with open(pickleName, 'wb') as fid:
        pickle.dump(fio, fid, protocol=pickle.HIGHEST_PROTOCOL)
    return fio

def FunwaveAnalyze(startTime, inputDict, fio):
    """This runs the post process script for FUNWAVE simulations.
    
    The script will read model output, create plots, and netcdf files.

    Args:
        inputDict (dict): this is an input dictionary that was generated with the
            keys from the project input yaml file
        startTime (str): input start time with datestring in format YYYY-mm-ddThh:mm:ssZ


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    print("check docstrings for Analyze and preprocess")
    # ___________________define Global Variables__________________________________

    plotFlag = inputDict.get('plotFlag', True)
    version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()
    Thredds_Base = inputDict.get('netCDFdir', '/thredds_data')
    server = inputDict.get('THREDDS', 'CHL')
    # the below should error if not included in input Dict
    path_prefix = inputDict['path_prefix']  # for organizing data
    simulationDuration = inputDict['simulationDuration']
    model = inputDict.get('modelName', 'SWASH').lower()
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, datestring)

    #_____________________________________________________________________________
    #_____________________________________________________________________________

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    prepdata = STPD.PrepDataTools()         # initializing instance for rotation scheme
    SeaSwellCutoff = 0.05
    nSubSample = 5                          # data are output at high rate, how often do we want to plot

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    matfile = os.path.join(fpath, ''.join(fio.ofileNameBase.split('-')) + '.mat')
    print('Loading files ')
    simData, simMeta = fio.loadSwash_Mat(fname=matfile)  # load all files
    ######################################################################################################################
    #################################   obtain total water level   #######################################################
    ######################################################################################################################
    #TODO: @Gaby, we'll use chuans/mine runup ID code here and save the runup time series data.
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
    if not os.path.exists(os.path.join(path_prefix,datestring, 'figures')):
        os.makedirs(os.path.join(path_prefix,datestring, 'figures'))  # make the directory for the simulation plots
    figureBaseFname = 'CMTB_waveModels_{}_{}_'.format(model, version_prefix)
    from matplotlib import pyplot as plt
    # make function for processing timeseries data
    # TODO: @Gaby, these should look familiar!
    fspec, freqs = sbwave.timeSeriesAnalysis1D(simData['time'].squeeze(), simData['eta'].squeeze(), bandAvg=6)
    total = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=None)
    SeaSwellStats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
    IGstats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)
    HsTS = 4 * np.std(simData['eta'].squeeze(), axis=0)

    #############################################################################################################
    ####################################### loop over tS plt ####################################################
    #############################################################################################################
    setup = np.mean(simData['eta'], axis=0).squeeze()
    WL = simMeta['WL'] #added in editing, should possibly be changed?
    if plotFlag == True:
        #TODO: @gaby, here  we'll be making QA/QC plots
        from plotting import operationalPlots as oP
        ## remove images before making them if reprocessing
        imgList = glob.glob(os.path.join(path_prefix, datestring, 'figures', '*.png'))
        [os.remove(ff) for ff in imgList]
        tstart = DT.datetime.now()

        ############### write a parallel data load function ##################
        dataOut = []
        def parallel_generateCrossShoreTimeSeries(tidx):
            ## generate a function that operates with only one input, can access local variable space
            timeStep = simData['time'][tidx]
            ofPlotName = os.path.join(path_prefix, datestring, 'figures',
                                      figureBaseFname + 'TS_' + timeStep.strftime('%Y%m%dT%H%M%S%fZ') + '.png')
            oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'],
                                             simData['xFRF'])
            dataOut.append(ofPlotName)
        ############### make TS plot in parallel -- has bugs   #########################################################
        #nprocessors = multiprocessing.cpu_count()/2                  # process plots with half the number on the
        # machine
        # pool = multiprocessing.Pool(nprocessors)                   # open multiprocessing pool
        # _ = pool.map(parallel_generateCrossShoreTimeSeries, range(0, len(simData['time']), nSubSample))
        # pool.close()
        #
        # print('Took {} long to make all the plots in parallel {} processors'.format(DT.datetime.now() - tstart, nprocessors))
        # ### now make gif of waves moving across shore*
        # imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '*TS_*.png')))
        # sb.makegif(imgList,
        #            os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.gif'.format(datestring)),
        #            dt=0.1)
        # print('Took {} long to make the movie and all the plots '.format(DT.datetime.now() - tstart))
        #$$$$$$ in Seriel $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for tidx in np.arange(0, len(simData['time']), nSubSample).astype(int):
            ofPlotName = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_' + simData['time'][tidx].strftime('%Y%m%dT%H%M%S%fZ') +'.png')
            oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'], simData['xFRF'])
        # now make gif of waves moving across shore
        imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', '*_TS_*.png')))
        dt = np.median(np.diff(simData['time'])).microseconds / 1000000
        sb.makeMovie(os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.avi'.format(datestring)), imgList, fps=nSubSample*dt)
        tarOutFile = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS.tar.gz')
        sb.myTarMaker(tarOutFile, imgList)

        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TempFname.png')
        oP.plotCrossShoreSummaryTS(ofname, simData['xFRF'], simData['elevation'], total,
                               SeaSwellStats, IGstats, setup=setup, WL=WL)
        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_spectrograph.png')
        oP.crossShoreSpectrograph(ofname, simData['xFRF'], freqs, fspec)
        ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + '_surfaceTimeseries.png')
        oP.crossShoreSurfaceTS2D(ofname, simData['eta'], simData['xFRF'], simData['time'])

    ##################################################################################################################
    ######################        Make NETCDF files       ############################################################
    ##################################################################################################################
    ##################################################################################################################

    #TODO: @Gaby, the last st   ep, we'll be making netCDF files.  I'd like to loop matt and ty and maybe mike in here
    # as we're going to be doing this for the LAB and it'd be nice to establish the "correct" format right off the
    # bat here for FUNWAVE, then they can absorb what we generate to implement directly into the model. we have a
    # tool to make netCDF files and it basically works by taking the model output and putting it into a dictionary.
    # That dictionary will match the file: yaml_files/waveModels/funwave/funwave_var.yml hand the data to the
    # function and the global metadata (yaml_files/waveModels/funwave/base/funwave_global.yml) and poof, it makes a
    # matching netCDF file!  This will be the end of the workflow.
    tsTime = np.arange(0, len(simData['time'])*dt, dt)

    fldrArch = os.path.join(model, version_prefix)
    spatial = {'time': nc.date2num(d1, units='seconds since 1970-01-01 00:00:00'),
               'station_name': '{} Field Data'.format(model),
               'tsTime': tsTime,
               'waveHsIG': np.reshape(IGstats['Hm0'], (1, len(simData['xFRF']))),
               'eta': np.swapaxes(simData['eta'], 0, 1),
               'totalWaterLevel': maxRunup,
               'totalWaterLevelTS': np.reshape(runup, (1, len(runup))),
               'velocityU': np.swapaxes(simData['velocityU'], 0, 1),
               'velocityV': np.swapaxes(simData['velocityV'], 0, 1),
               'waveHs': np.reshape(SeaSwellStats['Hm0'], (1, len(simData['xFRF']))), # or from HsTS??
               'xFRF': simData['xFRF'],
               'yFRF': simData['yFRF'][0],
               'runTime': np.expand_dims(fio.simulationWallTime, axis=0),
               'nProcess': np.expand_dims(fio.nprocess, axis=0),
               'DX': np.median(np.diff(simData['xFRF'])).astype(int),
               'DY': 1,    # must be adjusted for 2D simulations
               'NI': len(simData['xFRF']),
               'NJ': simData['velocityU'].shape[1],}  # should automatically adjust for 2D simulations

    TdsFldrBase = os.path.join(Thredds_Base, fldrArch)
    NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Field'), datestring, model=model)
    # make the name of this nc file
    NCname = 'CMTB-waveModels_{}_{}_Field_{}.nc'.format(model, version_prefix, datestring)
    fieldOfname = os.path.join(NCpath, NCname)

    if not os.path.exists(TdsFldrBase):
        os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    if not os.path.exists(os.path.join(TdsFldrBase, 'Field', 'Field.ncml')):
        inputOutput.makencml(os.path.join(TdsFldrBase, 'Field', 'Field.ncml'))  # remake the ncml if its not there
    # make file name strings
    flagfname = os.path.join(fpath, 'Flags{}.out.txt'.format(datestring))  # startTime # the name of flag file
    fieldYaml = 'yaml_files/waveModels/{}/{}_global.yml'.format(model, model)  # field
    varYaml = 'yaml_files/waveModels/{}/{}_var.yml'.format(model, model)
    assert os.path.isfile(fieldYaml), 'NetCDF yaml files are not created'  # make sure yaml file is in place
    makenc.makenc_phaseresolved(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)