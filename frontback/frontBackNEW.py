# -*- coding: utf-8 -*-
"""Pre and post processing associated with ww3 model runs."""
#from prepdata import inputOutput
from prepdata.prepDataLib import PrepDataTools
from getdatatestbed.getDataFRF import getDataTestBed
import datetime as DT
import os, glob, makenc, sys, shutil
from subprocess import check_output
import netCDF4 as nc
import numpy as np
import plotting.operationalPlots as oP
from testbedutils import sblib as sb
from testbedutils import waveLib
from testbedutils import fileHandling
from prepdata import py2netCDF as p2nc
from prepdata import postData

def ww3simSetup(startTimeString, inputDict, allWind , allWL, allWave, wrr):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) for ww3.

    Designed for unstructured grids (wave)

    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection
    
    Args:
        startTime (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function
        allWind(dict): from get data with all wind from entire project set
        allWL(dict): from get data with all waterlevel from entire project set
        allWave(dict): from get data with all wave from entire project set

    """
    # begin by setting up input parameters
    simulationDuration = int(inputDict.get('simulationDuration', 24))
    plotFlag = inputDict.get('plotFlag', True)
    

    print('TODO: rename these unpacked variables [frontbackNew.preprocess]')
    try:
        version_prefix = wrr.versionPrefix
    except:
        version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()

    plotFlag = inputDict.get('plotFlag', True)
    pathPrefix = wrr.workingDirectory
    model = wrr.modelName  # inputDict['modelSettings'].get('model', 'ww3').lower()
    rawspec = allWave
    rawwind = allWind
    rawWL = allWL
    
    # ___________________define version parameters_________________________________
    full = True
    # __________________set times _________________________________________________
    startTime = DT.datetime.strptime(startTimeString, '%Y-%m-%dT%H:%M:%SZ')
    endTime = startTime + DT.timedelta(0, simulationDuration * 3600, 0)
    dateString = wrr.dateString  # startTime.strftime('%Y-%m-%dT%H%M%SZ')

    print("Model Time Start : %s  Model Time End:  %s" % (startTime, endTime))

    # __________________Make Diretories_____________________________________________
    fileHandling.makeCMTBfileStructure(pathPrefix)

    # ____________________________ begin model data gathering __________________________________________________________
    gdTB = getDataTestBed(startTime, endTime)        # initialize get data test bed (bathy)
    prepdata = PrepDataTools()

    # _____________________WAVES, wind, WL ____________________________
    assert rawspec is not None, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (startTime, endTime)
    # use generated time lists for these to provide accurate temporal values
    _, waveTimeList, wlTimeList, _, windTimeList = prepdata.createDifferentTimeLists(startTime, endTime, rawspec, rawWL,
                                                                                     rawWind=rawwind)
    nFreq = np.size(rawspec['wavefreqbin'])
    # rotate and lower resolution of directional wave spectra
    wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=dateString, plot=plotFlag, full=full, deltaangle=10,
                                    outputPath=pathPrefix, model=model, waveTimeList=waveTimeList, ww3nFreq=nFreq)
    print('TODO: @Ty add values for nFreq here! [frontBackNew.line72]')
    windpacket = prepdata.prep_wind(rawwind, windTimeList, model=model)  # vector average, rotate winds, correct to 10m
    wlpacket = prepdata.prep_WL(rawWL, wlTimeList)                       # scalar average WL

    # ____________ BATHY   _____________________________________________
    bathy = gdTB.getBathyIntegratedTransect(method=1)
    gridNodes = wrr.readWW3_msh(fname=inputDict['modelSettings']['grid'])
    # gridNodes = sb.Bunch({'points': ww3io.points})              # we will remove this when meshio is working as expected

    if plotFlag: bathyPlotFname = os.path.join(pathPrefix, 'figures', dateString+'_bathy.png');
    else: bathyPlotFname=False
    bathy = prepdata.prep_Bathy(bathy, gridNodes, unstructured=True, plotFname=bathyPlotFname)

    # ____________________________ set model save points _______________________________________________________________
    # _________________________ Create observation locations ___________________________________________________________
    try:
        from testbedutils.frfTDSdataCrawler import query
        dataLocations = query(startTime, endTime, inputName=inputDict['TDSdatabase'],  type='waves')

    # # get gauge nodes x/y new idea: put gauges into input/output instance for the model, then we can save it
        gaugelocs = []
        for ii, gauge in enumerate(dataLocations['Sensor']):
             gaugelocs.append([dataLocations['Lon'][ii], dataLocations['Lat'][ii], gauge])
        wrr.savePoints = gaugelocs
    except:
        wrr.savePoints = [[-75.7428842,36.1872331,'8m-array']]
    
    # ____________________________ begin writing model Input ___________________________________________________________
    # ww3io.WL = WLpacket['avgWL']

    gridFname = inputDict['modelSettings']['grid']

    return wavepacket, windpacket, wlpacket, bathy, gridFname, wrr

def swashSimSetup(startTimeString, inputDict, allWind, allWL, allWave, wrr):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the Swash wave/FLow model


    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTimeString (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function
    Keyword Args:
        'bathyMethod': 0 for closet in time, 1 closest in history -- why is bathy retrieved in this function

    """
    # begin by setting up input parameters
    runtime = inputDict['modelSettings'].get('runDuration', 30*60)
    dx = inputDict['modelSettings'].get('dx', 1)
    dy = inputDict['modelSettings'].get('dy', 1)
    yBounds = inputDict['modelSettings'].get('yBounds', [944, 947])
    nf = inputDict['modelSettings'].get('nf', 200)
    bathyMethod = inputDict.get('bathyMethod', 1)
    version_prefix = wrr.versionPrefix
    model = wrr.modelName
    rawspec = allWave
    rawWL = allWL
    del allWind  # to take care of all inputs
    # ______________________________________________________________________________
    
    # _______________________________________________________________________________
    # set times
    d1 = DT.datetime.strptime(startTimeString, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, runtime, 0)
    
    fileHandling.checkVersionPrefix(model=wrr.modelName, inputDict=inputDict)
    fileHandling.displayStartInfo(d1, d2, wrr.workingDirectory, None, wrr.modelName)
    fileHandling.makeCMTBfileStructure(wrr.workingDirectory)

    # ______________________________________________________________________________
    # begin model data gathering

    prepdata = PrepDataTools()                      # for preprocessing
    gdTB = getDataTestBed(d1, d2)        # for bathy data gathering
    print('TODO: discussion - currently grabbing bathy in simSetup -- should i be? [frontBackNew.swashSimSetup]')
    # _____________WAVES____________________________
    
    # preprocess wave spectra
    # idxSim = np.argmin(np.abs(DT.datetime.strptime(startTimeString, '%Y-%m-%dT%H:%M:%SZ') - rawspec['time']))
    # rawspec = sb.reduceDict(rawspec, idxSim)  # keep only a single simulation
    wavepacket = prepdata.prep_spec_phaseResolved(rawspec, version_prefix, runDuration=runtime, nf=nf,
                                                  waveTimeList=DT.datetime.strptime(wrr.dateString, wrr.dateStringFmt))

    ## ___________WATER LEVEL__________________
    WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])
    ### ____________ Get bathy grid from thredds ________________
    bathy = gdTB.getBathyIntegratedTransect(method=bathyMethod, ybounds=yBounds, type='bathyTopo')
    gridDict = prepdata.prep_SwashBathy(wavepacket['xFRF'], wavepacket['yFRF'], bathy, dx=dx, dy=dy,
                                                 yBounds=yBounds)  # non-inclusive index if you want 3 make 4 wide
    
    return wavepacket, None, WLpacket, gridDict, None, wrr

def genericPostProcess(startTime, inputDict, spatialData, pointData, wrr):
    """This runs the post process script for Wave Watch 3.

     Script will load model output, create netCDF files and make plots of output including basic model-data
     comparison.

    Args:
        inputDict (dict): this is an input dictionary that was generated with the
            keys from the project input yaml file
        startTime (str): input start time with datestring in format YYYY-mm-ddThh:mm:ssZ
        wrr(class): this is an initialized class from model pre-processing.  This holds specific metadata including
            class objects as listed below:


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    # ___________________define Global Variables___________________________________
    plotFlag = inputDict.get('plotFlag', True)
    model = wrr.modelName
    path_prefix = wrr.workingDirectory
    # simulationDuration = inputDict['simulationDuration']
    Thredds_Base = inputDict.get('netCDFdir', wrr.workingDirectory)
    # version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()
    # grid = inputDict['modelSettings'].get('grid').lower()
    # endTime = inputDict['endTime']
    # startTime = inputDict['startTime']
    # simulationDuration = inputDict['simulationDuration']
    # workingDir = inputDict['workingDirectory']
    # generateFlag = inputDict['generateFlag']
    # runFlag = inputDict['runFlag']
    # analyzeFlag = inputDict['analyzeFlag']
    # plotFlag = inputDict['plotFlag']
    # model = inputDict.get('modelName', 'FUNWAVE').lower()
    # inputDict['path_prefix'] = os.path.join(workingDir, model, version_prefix)
    #
    # path_prefix = inputDict['path_prefix']
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    # try:
    #projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    #projectEnd = DT.datetime.strptime(inputDict['endTime'], '%Y-%m-%dT%H:%M:%SZ')
    dateString = wrr.dateString  # a string for file names

    # _____________________________________________________________________________
    version_prefix = fileHandling.checkVersionPrefix(model, inputDict)
    # define based on model name what postprocessing switches need to be turned on/off
    postProcessingDict = fileHandling.modelPostProcessCheck(model)
    
    # __________________________1. post process ____________________________________
    if postProcessingDict['phaseResolved'] == True:
        spatialPostProcessed = postData.processNLwave(spatialData, wrr=wrr, ncFilesOnly=False)
        pointPostProcessed = None
    else:
        spatialPostProcessed, pointPostProcessed = someFunctionPostProcesing(spatialData, pointData)
    
    # __________________________2. make netCDF Files________________________________
    # __________________________2a. Spatial output _________________________________

    if spatialPostProcessed is not None:
        fldrArch = os.path.join(model, version_prefix)
        fieldYaml = f'yaml_files/waveModels/{fldrArch}/{wrr.modelName}_global.yml'
        varYaml = f'yaml_files/waveModels/{wrr.modelName}/{wrr.modelName}_var.yml'
        fieldOfname = fileHandling.makeTDSfileStructure(Thredds_Base, fldrArch, dateString, 'Field')
        p2nc.makenc_generic(fieldOfname, globalYaml=fieldYaml, varYaml=varYaml, data=spatialPostProcessed)
    # __________________________2aa. Spatial plotting ________________________________
        ncfile = nc.Dataset(fieldOfname)
        if postProcessingDict['phaseResolved'] == True:
            # remove any plots in the plotting folder first
            [os.remove(ff) for ff in glob.glob(wrr.plottingDirectory +'*.png')]
            oP.makeCrossShoreTimeSeriesPlotAndMovie(ncfile, plottingDirectory=wrr.plottingDirectory,
                                                     versionPrefix=wrr.versionPrefix, xBounds=[0,250], figsize=(6,3))
            ofname = os.path.join(wrr.plottingDirectory, f'CMTB_{wrr.modelName}_'
                                                         f'{wrr.versionPrefix}_CrossShoreSummary.png')
            oP.plotCrossShoreSummaryTS(ofname, ncfile, Hs_ss=spatialPostProcessed['X_statsSS']['Hm0'], WL=np.nan)
            ofname = os.path.join(wrr.plottingDirectory, f'CMTB_{wrr.modelName}_{wrr.versionPrefix}_spectrograph.png')
            
            oP.crossShoreSpectrograph(ofname, ncfile['xFRF'][:].squeeze(), ncfile['waveFrequency'][:].squeeze(),
                                      ncfile['waveEnergyDensity'][:].squeeze())
            ofname = os.path.join(wrr.plottingDirectory, f'CMTB_{wrr.modelName}_'
                                                         f'{wrr.versionPrefix}_2DsurfaceTimeSeries.png')
            oP.crossShoreSurfaceTS2D(ofname, ncfile['eta'][:].squeeze(), ncfile['xFRF'][:].squeeze(), ncfile['tsTime'][
                                                                                                      :].squeeze())

        elif postProcessingDict['phaseAveraged'] is True:
            variables = ncfile.variables.keys()
            for var in variables:
                if var in ['waveHs', 'waveDm', 'waveTm'] and plotFlag is True:
                    plotOutFname = 'test.png'
                    oP.unstructuredSpatialPlot(plotOutFname, fieldNc=ncfile, variable=var)
        else:
            raise NotImplementedError("other model plotting types haven't been implemented yet")
    # __________________________2b. point output _________________________________
    if pointPostProcessed is not None:
        print(' post-processing point data is not developed yet')
        fldrArch = os.path.join(model, version_prefix)
        varYaml_fname = f'yaml_files/waveModels/{fldrArch}/Station_var.yml'
        globalyaml_fname = f'yaml_files/waveModels/{fldrArch}/Station_globalmeta.yml'
        # frame work is shown below, post processing lays out nice nested dictionary for every save station
        for ss, station in enumerate(pointPostProcessed['stationList']):
            outFileName = fileHandling.makeTDSfileStructure(Thredds_Base, fldrArch, dateString, station)
            p2nc.makenc_generic(outFileName, globalyaml_fname, varYaml_fname, pointPostProcessed[ss])
    # __________________________2bb. point plotting ________________________________
        print("is this necessary??")
        

   

   

def cshoreSimSetup(startTimeString, inputDict, allWave, allBathy, allWL, allWind, allCTD, wrr):
    """Author: David Young
    Association: USACE CHL Field Research Facility
    Project:  Coastal Model Test Bed

    This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB).  It is designed to pull from GetData and utilize
    prep_datalib for development of the FRF CMTB
    NOTE: input to the function is the start time of the model run.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): this is the start time for the simulation (string in format e.g., '2016-06-02T10:00:00Z' )
                THIS MAY NOT BE THE SAME AS THE ONE IN INPUT DICT
                i.e., if it is looping over a bunch of 24 hour simulations
                that is also why it is a separate variable
        inputDict (dict): input dictionary with keys
            simulationDuration - duration of each simulation in hours
            version_prefix - right now we have FIXED, MOBILE, and MOBILE_RESET
            profileNumber - this is either the survery profile number or the alongshore location for the integrated bathy
            bathyLoc - where are we getting our bathy data (surveys or integrated bathy)
            workindDir - location where the user wants to have all the data and stuff

    """
    # begin by setting up input parameters
    simulationDuration = int(inputDict.get('simulationDuration', 24))
    timeStep = (inputDict.get('timeStep',3600))
    plotFlag = inputDict.get('plotFlag', True)

    print('TODO: rename these unpacked variables [frontbackNew.preprocess]')
    try:
        version_prefix = wrr.versionPrefix
    except:
        version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()

    plotFlag = inputDict.get('plotFlag', True)
    pathPrefix = wrr.workingDirectory
    model = wrr.modelName  # inputDict['modelSettings'].get('model', 'ww3').lower()
    rawspec = allWave
    rawwind = allWind
    rawWL = allWL
    rawCTD = allCTD
    rawBathy = allBathy


    # ___________________define version parameters_________________________________
    full = True
    # __________________set times _________________________________________________
    startTime = DT.datetime.strptime(startTimeString, '%Y-%m-%dT%H:%M:%SZ')
    endTime = startTime + DT.timedelta(0, simulationDuration * 3600, 0)
    dateString = wrr.dateString  # startTime.strftime('%Y-%m-%dT%H%M%SZ')
    ftime = simulationDuration*3600.
    reltime = np.arange(0,ftime+timeStep,timeStep)
    print("Model Time Start : %s  Model Time End:  %s" % (startTime, endTime))

    # __________________Make Diretories_____________________________________________
    fileHandling.makeCMTBfileStructure(pathPrefix)

    # ____________________________ begin model data gathering __________________________________________________________
#    gdTB = getDataTestBed(startTime, endTime)  # initialize get data test bed (bathy)
    prepdata = PrepDataTools()

    # _____________________WAVES, wind, WL ____________________________
    assert rawspec is not None, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (startTime, endTime)
    # use generated time lists for these to provide accurate temporal values
 #   _, waveTimeList, wlTimeList, _, windTimeList = prepdata.createDifferentTimeLists(startTime, endTime, rawspec, rawWL,
#                                                                                     rawWind=rawwind)
 #   nFreq = np.size(rawspec['wavefreqbin'])
    # rotate and lower resolution of directional wave spectra
 #   wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=dateString, plot=plotFlag, full=full, deltaangle=5,
 #                                   outputPath=pathPrefix, model=model, waveTimeList=waveTimeList, ww3nFreq=nFreq)
 #   print('TODO: @Ty add values for nFreq here! [frontBackNew.line72]')
    print(reltime, startTime)
    wavepacket = prepdata.prep_CSHOREwaves(rawspec, reltime , startTime)

    wlTimeList = [startTime + DT.timedelta(seconds=tt) for tt in reltime]
    WLpacket = prepdata.prep_WL(rawWL,wlTimeList)
    windTimeList= [startTime + DT.timedelta(seconds=tt) for tt in reltime]
    windpacket = prepdata.prep_wind(rawwind, windTimeList, model=model)  # vector average, rotate winds, correct to 10m

    # pull the stuff I need out of the dict
    timerun = inputDict['simulationDuration']
    version_prefix = inputDict['modelSettings']['version_prefix']
    profile_num = inputDict['profileNumber']
    bathy_loc = inputDict['bathyLoc']
    workingDir = inputDict['workingDirectory']
    if 'THREDDS' in inputDict:
        server = inputDict['THREDDS']
    else:
        print('Chosing CHL thredds by Default, this may be slower!')
        server = 'CHL'
    # ____________________GENERAL ASSUMPTION VARIABLES__________

    model = 'CSHORE'
    path_prefix = os.path.join(workingDir, model,  '%s/' % version_prefix)
    time_step = 1        # time step for model in hours
    dx = 1               # cross-shore grid spacing (FRF coord units - m)
    fric_fac = 0.015     # default friction factor

    # ______________________________________________________________________________
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # Time Stuff!
    if type(timerun) == str:
        timerun = int(timerun)
    #start_time = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    start_time = startTime
    bathy_loc_List = np.array(['integrated_bathy', 'survey'])

    assert start_time.minute == 0 and start_time.second == 0 and start_time.microsecond == 0, 'Your simulation must start on the hour!'

    end_time = start_time + DT.timedelta(days=0, hours=timerun) # removed for ilab=1 , minutes=1)
    print(end_time)
    date_str = start_time.strftime('%Y-%m-%dT%H%M%SZ')
    # start making my metadata dict
    #meta_dict = {'startTime': DT.datetime.strftime(startTime,'%Y-%m-%dT%H:%M:%SZ'),
    #             'timerun': timerun,
    #             'time_step': time_step,
    #             'dx': dx,
    #             'fric_fac': fric_fac,
    #             'version': version_prefix}
    #ftime = timerun * 3600  # [sec] final time, dictates model duration
    #dt = time_step * 3600  # time interval (sec) for wave and water level conditions
    #BC_dict = {'timebc_wave': np.arange(0, ftime + dt, dt)}

    bathypacket = prepdata.prep_CSHOREbathy(rawBathy, bathy_loc, dx, wavepacket, profile_num=profile_num,fric_fac=fric_fac)
    # ______________________________________________________________________________
    # __________________Make Diretories_____________________________________________
    #if not os.path.exists(path_prefix + date_str):  # if it doesn't exist
    #    os.makedirs(path_prefix + date_str)  # make the directory
    #if not os.path.exists(path_prefix + date_str + "/figures/"):
    #    os.makedirs(path_prefix + date_str + "/figures/")

    print("Model Time Start : %s  Model Time End:  %s" % (start_time, end_time))
    #print("Files will be placed in {0} folder".format(path_prefix + date_str))

    return wavepacket, windpacket, WLpacket, bathypacket, rawCTD, wrr

