"""Pre and post processing associated with ww3 model runs."""
from prepdata import inputOutput
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
    wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=dateString, plot=plotFlag, full=full, deltaangle=5,
                                    outputPath=pathPrefix, model=model, waveTimeList=waveTimeList, ww3nFreq=nFreq)
    print('TODO: @Ty add values for nFreq here! [frontBackNew.line72]')
    
    windpacket = prepdata.prep_wind(rawwind, windTimeList, model=model)  # vector average, rotate winds, correct to 10m
    WLpacket = prepdata.prep_WL(rawWL, wlTimeList)                       # scalar average WL

    # ____________ BATHY   _____________________________________________
    bathy = gdTB.getBathyIntegratedTransect(method=1)
    gridNodes = wrr.readWW3_msh(inputDict['modelSettings']['grid'])
    # gridNodes = sb.Bunch({'points': ww3io.points})              # we will remove this when meshio is working as expected

    if plotFlag: bathyPlotFname = os.path.join(pathPrefix, 'figures', dateString+'_bathy.png');
    else: bathyPlotFname=False
    bathy = prepdata.prep_Bathy(bathy, gridNodes, unstructured=True, plotFname=bathyPlotFname)

    # ____________________________ set model save points _______________________________________________________________
    # _________________________ Create observation locations ___________________________________________________________
    from testbedutils.frfTDSdataCrawler import query
    dataLocations = query(startTime, endTime, inputName=inputDict['TDSdatabase'],  type='waves')

    # # get gauge nodes x/y new idea: put gauges into input/output instance for the model, then we can save it
    gaugelocs = []
    for ii, gauge in enumerate(dataLocations['Sensor']):
         gaugelocs.append([dataLocations['Lon'][ii], dataLocations['Lat'][ii], gauge])
    wrr.savePoints = gaugelocs
    
    # ____________________________ begin writing model Input ___________________________________________________________
    # ww3io.WL = WLpacket['avgWL']

    gridFname = inputDict['modelSettings']['grid']

    return wavepacket, windpacket, WLpacket, bathy, gridFname, wrr

def swashSimSetup(startTimeString, inputDict, allWind, allWL, allWave, wrr):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the Swash wave/FLow model


    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTimeString (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function

    """
    # begin by setting up input parameters
    runtime = inputDict.get('simulationDuration', 30*60)
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
    # _____________WAVES____________________________
    
    # preprocess wave spectra
    wavepacket = prepdata.prep_spec_phaseResolved(rawspec, version_prefix, runDuration=runtime*60*60,
                                                  waveTimeList=DT.datetime.strptime(wrr.dateString, wrr.dateStringFmt))
    ## ___________WATER LEVEL__________________
    WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])
    ### ____________ Get bathy grid from thredds ________________
    bathy = gdTB.getBathyIntegratedTransect(method=1, ybounds=[940, 950])
    gridDict = prepdata.prep_SwashBathy(wavepacket['xFRF'], wavepacket['yFRF'], bathy, dx=1, dy=1,
                                                 yBounds=[944, 947])  # non-inclusive index if you want 3 make 4 wide
    
    return wavepacket, None, WLpacket, gridDict, None, wrr
    

def genericPostProcess(startTime, inputDict, ww3io):
    """This runs the post process script for Wave Watch 3.

     Script will load model output, create netCDF files and make plots of output including basic model-data
     comparison.

    Args:
        inputDict (dict): this is an input dictionary that was generated with the
            keys from the project input yaml file
        startTime (str): input start time with datestring in format YYYY-mm-ddThh:mm:ssZ
        ww3io(class): this is an initialized class from model pre-processing.  This holds specific metadata including
            class objects as listed below:


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    # ___________________define Global Variables___________________________________
    plotFlag = inputDict.get('plotFlag', True)
    model = inputDict.get('model', 'ww3')
    version_prefix = ww3io.version_prefix
    path_prefix = inputDict.get(os.path.join('path_prefix', "{}".format(version_prefix)), ww3io.path_prefix)
    simulationDuration = inputDict['simulationDuration']
    Thredds_Base = inputDict.get('netCDFdir', '/home/{}/thredds_data/'.format(check_output('whoami', shell=True)[:-1]))

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    # try:
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    dateString = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, dateString)

    # ____________________________________________________________________________
    if version_prefix == 'base':
        full = True   # define full plane
    # _____________________________________________________________________________
    print('\nBeginning of Analyze Script\nLooking for file in {}'.format(simulationDirectory))
    print('\nData Start: {}  Finish: {}'.format(d1, d2))
    print('Analyzing simulation')
    # go = getDataFRF.getObs(d1, d2, server)  # setting up get data instance
    prepdata = PrepDataTools()  # initializing instance for rotation scheme
    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    t = DT.datetime.now()
    print('Loading files ')
    savePointFname = os.path.join(ww3io.path_prefix, "ww3.{}_spec.nc".format(DT.datetime.strptime(ww3io.dateString,
                                                                                  '%Y-%m-%dT%H%M%SZ').strftime('%Y%m')))
    pointNc = nc.Dataset(savePointFname)                                                         # load point file
    fieldNc = ww3io.readWW3_field()                                                              # load all files
    bathyPacket = ww3io.readWW3_msh()                                                            # load bathy as input
    print('Loaded files in {:.2f} seconds'.format((DT.datetime.now() - t).total_seconds()/60))

    ##### point file processing
    idxSorted = np.argsort(pointNc['direction'][:])
    dWEDout = pointNc['efth'][:, :, :, idxSorted]

    for ss, station in enumerate(nc.chartostring(pointNc['station_name'][:])):
        stats = waveLib.waveStat(dWEDout[:,1],frqbins=pointNc['frequency'][:], dirbins=pointNc['direction'][:])
        out = {'waveHs': stats['Hm0'],
               'time': nc.date2num(nc.num2date(pointNc['time'][:], pointNc['time'].units), 'seconds since 1970-01-01'),
               'waveTm': stats['Tm'],
               'waveDm': stats['Dm'],
               'waveTp': stats['Tp'],
               'waterLevel': np.ones_like(stats['Hm0']) * -999,
               'windSpeed': pointNc['wnd'][:, ss],
               'windDirection': pointNc['wnddir'][:, ss],
               'currentSpeed': pointNc['cur'][:, ss],
               'currentDirection': pointNc['curdir'][:, ss],
               'flags': np.ones_like(stats['Hm0']) * -999,
               'longitude': pointNc['longitude'][0, ss],        # assume static location
               'latitude': pointNc['latitude'][0, ss],          # assume static location
               'waveDirectionBins': pointNc['direction'][:],
               'waveFrequency': pointNc['frequency'][:],
               'directionalWaveEnergyDensity': dWEDout[:,ss],
               'station_name': station
               }
        fldrArch = os.path.join(model, version_prefix)
        varYaml_fname = 'yaml_files/waveModels/{}/Station_var.yml'.format(fldrArch)
        globalyaml_fname = 'yaml_files/waveModels/{}/Station_globalmeta.yml'.format(fldrArch)
        outFileName = fileHandling.makeTDSfileStructure(Thredds_Base, fldrArch, datestring, station)
        p2nc.makenc_generic(outFileName, globalyaml_fname, varYaml_fname, out)

    ##### Field file processing
    out = {'time':            nc.date2num(nc.num2date(fieldNc['time'][:], fieldNc['time'].units), 'seconds since 1970-01-01'),
           'latitude':        fieldNc['latitude'][:],
           'longitude':       fieldNc['longitude'][:],
           'meshName':        -999,
           'connectivity':    fieldNc['tri'][:],
           'three':           np.ones((3)) * -999,
           'nfaces':          np.arange(fieldNc['tri'].shape[0], dtype=int),
           'nnodes':          np.arange(fieldNc.dimensions['node'].size, dtype=int),
           'xFRF':            np.ones_like(fieldNc['latitude'][:]) * -999,
           'yFRF':            np.ones_like(fieldNc['latitude'][:]) * -999,
           'waveHs':          np.ma.masked_array(fieldNc['hs'][:], mask=fieldNc['hs']._FillValue),
           'bathymetry':      bathyPacket['points'][:,2],  # doesn't need to be expanded into time dimension
           'waveTm':          fieldNc['t02'][:],
           'waveDm':          fieldNc['dir'][:],
           'dynamicTimeStep': fieldNc['dtd'][:] * 60,  # convert to seconds
           'mapStatus':       fieldNc['MAPSTA'][:]
           }
    fieldYaml = 'yaml_files/waveModels/%s/Field_globalmeta.yml' % (fldrArch)  # field
    varYaml = 'yaml_files/waveModels/%s/Field_var.yml' % (fldrArch)
    fieldOfname = fileHandling.makeTDSfileStructure(Thredds_Base, fldrArch, datestring, 'Field')
    p2nc.makenc_generic(fieldOfname, globalYaml=fieldYaml, varYaml=varYaml, data=out)
    
    
    ncfile = nc.Dataset(fieldOfname)
    variables = ncfile.variables.keys()
    for var in variables:
        if var in ['waveHs', 'waveDm', 'waveTm'] and plotFlag is True:
            plotOutFname = 'test.png'
            oP.unstructuredSpatialPlot(plotOutFname, fieldNc=ncfile, variable=var)
    

