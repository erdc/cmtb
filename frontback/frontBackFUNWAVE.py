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
    date_str = startTime #d1T%.strftime('%Y-%m-%dH%M%SZ')
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

    wavepacket = prepdata.prep_spec_phaseResolved(rawspec, version_prefix, model=model, nf=nf, phases=phases)

    # _____________WINDS______________________
    print('_________________\nSkipping Wind')
    
    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    WLpacket = prepdata.prep_WL(rawWL, rawWL['epochtime']) # time average WL
    # find WL corresponding to wanted date:
    WL_index = np.where(WLpacket['time']==wavepacket['time'])[0][0]
    WL = WLpacket['avgWL'][WL_index]

    ### ____________ Get bathy grid from thredds ________________

    if grid.lower() == '1d':    # non-inclusive index for yBounds
        ybounds = [bathy['yFRF']-1.5*dy,bathy['yFRF']+1.5*dy]# [bathy['yFRF']-dy, bathy['yFRF']+dy]  ## should take a
        # look at this
    else:
        ybounds = [600,1100]

    _, gridDict = prepdata.prep_SwashBathy(bathy['xFRF'][0], bathy['yFRF'], bathy.copy(), ybounds)  #

    # _____________ begin writing files _________________________
    # set some of the class instance variables before writing input files
    # TODO: @Gaby, calculate nprocessors (px * py), i think this is based on the grid, so you can use the output from
    #  prep_FunwaveBathy

    [Nglob,Mglob] = gridDict['elevation'].shape
    px = np.floor(Mglob / 150)
    if grid.lower() == '1d':
        py = 3
    else:
        py = np.floor(Nglob / 150)
    if px <48:
        px = 48
    nprocessors = px * py  # now calculated on init

    fio = funwaveIO(fileNameBase=date_str, path_prefix=path_prefix, version_prefix=version_prefix, WL=WL,
                    equilbTime=0, Hs=wavepacket['Hs'], Tp=1/wavepacket['peakf'], Dm=wavepacket['waveDm'],
                    px=px, py=py, nprocessors=nprocessors,Mglob=Mglob,Nglob=Nglob)

    ## write spectra, depth, and station files
    if grid.lower() == '1d':
        fio.Write_1D_Bathy(Dep=gridDict['elevation'], xFRF=gridDict['xFRF'], yFRF=gridDict['yFRF'])
        fio.Write_1D_Spectra_File(wavepacket)
    else:
        fio.Write_2D_Bathy(Dep=gridDict['elevation'], xFRF=gridDict['xFRF'], yFRF=gridDict['yFRF'])
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
    model = inputDict.get('modelName', 'funwave').lower()
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    #d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    #d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)

    d1 = DT.datetime.strptime(inputDict['startTime'], '%Y-%m-%dT%H:%M:%SZ')
    d2 = DT.datetime.strptime(inputDict['endTime'], '%Y-%m-%dT%H:%M:%SZ')
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = path_prefix #os.path.join(path_prefix, datestring)

    #_____________________________________________________________________________
    #_____________________________________________________________________________

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################

    outputFolder = os.path.join(fpath,fio.ofileNameBase,'output')
    print('Loading files ',outputFolder)

    ## upload depth file
    depthFile = os.path.join(outputFolder, 'dep.out')
    if os.path.exists(depthFile)==True:
        try:
            Depth1D = fio.readasciidepthfile(depthFile)
        except:
            Depth1D = fio.readbinarydepthfile(depthFile)
    else:
        try:
            Depth1D = fio.readasciidepthfile(os.path.join(fpath,fio.ofileNameBase,'depth.txt'))
        except:
            Depth1D = fio.readbinarydepthfile(os.path.join(fpath,fio.ofileNameBase,'depth.txt'))

    simData, simMeta = fio.loadFUNWAVE_stations(Depth1D,fname=outputFolder)  # load all files

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
    fileHandling.makeCMTBfileStructure(path_prefix,date_str=datestring)
    figureBaseFname = 'CMTB_waveModels_{}_{}_'.format(model, version_prefix)

    # make function for processing timeseries data
    # TODO: @Gaby, these should look familiar! ... yup, It does XD
    cutRampingTime = 1200 # which equals 600sec (10min) for dt = 0.5sec
    data = simData['eta'].squeeze()[cutRampingTime:,:]

    time = []
    for i in range(len(simData['time'].squeeze()[cutRampingTime:])): ## change time from float to datetime
        dt_i = DT.timedelta(seconds =simData['time'].squeeze()[cutRampingTime:][i])
        time.append(d1+dt_i)

    SeaSwellCutoff = 0.05 # cutoff between sea/swell and IG
    nSubSample = 5

    fspec, freqs = sbwave.timeSeriesAnalysis1D(np.asarray(time),data, bandAvg=3)#6,WindowLength=20)
    total = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=None)
    SeaSwellStats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=SeaSwellCutoff, highFreq=None)
    IGstats = sbwave.stats1D(fspec=fspec, frqbins=freqs, lowFreq=None, highFreq=SeaSwellCutoff)
    HsTS = 4 * np.std(data, axis=0)

    #############################################################################################################
    ####################################### loop over tS plt ####################################################
    #############################################################################################################
    WL = simMeta['WL'] #added in editing, should possibly be changed?
    setup = np.mean(simData['eta'] + WL, axis=0).squeeze()
    if plotFlag == True:
        #TODO: @gaby, here  we'll be making QA/QC plots
        from plotting import operationalPlots as oP
        ## remove images before making them if reprocessing
        imgList = glob.glob(os.path.join(fpath,fio.ofileNameBase, 'figures', '*.png'))
        [os.remove(ff) for ff in imgList]
        tstart = DT.datetime.now()
        # TODO: write a parallel data plotting function
        #### in Seriel $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        for tidx in np.arange(0, len(simData['time']), nSubSample).astype(int):
            if tidx < np.shape(time)[0]:

                figPath = os.path.join(fpath,fio.ofileNameBase,'figures')
                ofPlotName = os.path.join(figPath, figureBaseFname + 'TS_' + time[tidx].strftime('%Y%m%dT%H%M%S%fZ') +'.png')

                bottomIn = -simData['elevation']
                dataIn = simData['eta'][tidx].squeeze()

                if np.median(bottomIn) > 0:
                    bottomIn = -bottomIn

                shoreline= np.where(dataIn > bottomIn)[0][0]
                dataIn[:shoreline] = float("NAN")

                oP.generate_CrossShoreTimeseries(ofPlotName, dataIn, bottomIn, simData['xFRF'])
        # now make gif of waves moving across shore
        imgList = sorted(glob.glob((os.path.join(figPath, '*_TS_*.png')))) #sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', '*_TS_*.png')))
        dt = np.median(np.diff(time)).microseconds / 1000000
        sb.makeMovie(os.path.join(figPath, figureBaseFname + 'TS_{}.avi'.format(datestring)), imgList, fps=nSubSample*dt)
        tarOutFile = os.path.join(figPath, figureBaseFname + 'TS.tar.gz')
        sb.myTarMaker(tarOutFile, imgList)

        ofname = os.path.join(figPath, figureBaseFname + 'crossShoreSummary.png')
        oP.plotCrossShoreSummaryTS(ofname, simData['xFRF'], simData['elevation'], total,
                               SeaSwellStats, IGstats, setup=setup, WL=WL)
        ofname = os.path.join(figPath, figureBaseFname + '_spectrograph.png')
        oP.crossShoreSpectrograph(ofname, simData['xFRF'], freqs, fspec)
        ofname = os.path.join(figPath, figureBaseFname + '_surfaceTimeseries.png')
        oP.crossShoreSurfaceTS2D(ofname, simData['eta'], simData['xFRF'], simData['time'])
        print("plotting took {} minutes".format((DT.datetime.now()-tstart).total_seconds()/60))
    ##################################################################################################################
    ######################        Make NETCDF files       ############################################################
    ##################################################################################################################
    ##################################################################################################################

    #TODO: @Gaby, the last step, we'll be making netCDF files.  I'd like to loop matt and ty and maybe mike in here
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
               'eta': simData['eta'],
               'totalWaterLevel': maxRunup,
               'totalWaterLevelTS': np.reshape(runup, (1, len(runup))),
               'velocityU': simData['velocityU'],
               'velocityV': simData['velocityV'],
               'waveHs': np.reshape(SeaSwellStats['Hm0'], (1, len(simData['xFRF']))),  # or from HsTS??
               'xFRF': simData['xFRF'],
               'yFRF': simData['yFRF'][0],
               'runTime': np.expand_dims(fio.simulationWallTime, axis=0),
               'nProcess': np.expand_dims(fio.nprocess, axis=0),
               'DX': np.expand_dims(fio.DX, axis=0),
               'DY': np.expand_dims(fio.DY, axis=0),  # must be adjusted for 2D simulations
               'NI': np.expand_dims(fio.Mglob, axis=0),
               'NJ': np.expand_dims(fio.Nglob, axis=0), }  # should automatically adjust for 2D simulations

    fieldOfname = fileHandling.makeTDSfileStructure(Thredds_Base, fldrArch, datestring, 'Field')
    # TdsFldrBase = os.path.join(Thredds_Base, fldrArch)
    # NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Field'), datestring, model=model)
    # # make the name of this nc file
    # NCname = 'CMTB-waveModels_{}_{}_Field_{}.nc'.format(model, version_prefix, datestring)
    # fieldOfname = os.path.join(NCpath, NCname)

    # if not os.path.exists(TdsFldrBase):
    #     os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    # if not os.path.exists(os.path.join(TdsFldrBase, 'Field', 'Field.ncml')):
    #     inputOutput.makencml(os.path.join(TdsFldrBase, 'Field', 'Field.ncml'))  # remake the ncml if its not there
    # # make file name strings
    flagfname = os.path.join(fpath, 'Flags{}.out.txt'.format(datestring))  # startTime # the name of flag file
    fieldYaml = 'yaml_files/waveModels/{}/{}/{}_global.yml'.format(model,version_prefix, model)  # field
    varYaml = 'yaml_files/waveModels/{}/{}_var.yml'.format(model, model)
    assert os.path.isfile(fieldYaml), 'NetCDF yaml files are not created'  # make sure yaml file is in place
    makenc.makenc_phaseresolved(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)
