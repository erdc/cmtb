"""
This script holds the master function for the simulation Setup
for the CMS wave/Flow? setup
"""
import prepdata.prepDataLib
import testbedutils.anglesLib
from prepdata import inputOutput
from getdatatestbed.getDataFRF import getDataTestBed
from getdatatestbed.getDataFRF import getObs
from getdatatestbed import getPlotData
import datetime as DT
import os, glob, shutil
from subprocess import check_output
import netCDF4 as nc
import numpy as np
import makenc
from prepdata import prepDataLib as STPD
from prepdata.inputOutput import cmsIO, stwaveIO, cmsfIO
from getdatatestbed import getDataFRF
import plotting.operationalPlots as oP
from testbedutils import sblib as sb
from testbedutils import waveLib as sbwave
from plotting.operationalPlots import obs_V_mod_TS
from testbedutils import gridTools as gT
from plotting import nonoperationalPlots as noP
import string

def CMSsimSetup(startTime, inputDict, **kwargs):
    """
    Author: Spicer Bak
    Association: USACE CHL Field Research Facility
    Project:  Coastal Model Test Bed

    This Function is the master call for the preprocessing for unstructured grid model runs
        it is designed to pull from GetData and utilize prep_datalib for development of the FRF CMTB

    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime(str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict(dict): this is a dictionary that is read from the yaml read function
         requires keys ['startTime', 'endTime', 'path_prefix',]  probably others (please add)

    """
    # begin by setting up input parameters
    timerun = inputDict.get('timerun', 24)
    pFlag = inputDict.get('pFlag', True)
    path_prefix = inputDict['path_prefix']
    model = inputDict.get('model', 'CMS').lower()
    version_prefix = inputDict.get('version_prefix', 'base').lower()

    # first up, need to check which parts I am running
    waveFlag = inputDict.get('wave', True)
    flowFlag = inputDict.get('flow', False)
    morphFlag = inputDict.get('morph', False)
    #TODO: check if better way to handle flags at begining of process


    if waveFlag:
        assert 'wave_version_prefix' in inputDict, 'Must have "wave_version_prefix" in your input yaml'
        wave_version_prefix = inputDict.get('wave_version_prefix', 'base').lower()
        version_prefix = version_prefix + '_' + wave_version_prefix
        # data check
        prefixList = np.array(['base','hp', 'untuned'])
        assert (wave_version_prefix.lower() == prefixList).any(), "Please enter a valid wave version prefix\n Prefix assigned = %s must be in List %s" % (wave_version_prefix, prefixList)
    if flowFlag:
        assert 'flow_version_prefix' in inputDict, 'Must have "flow_version_prefix" in your input yaml'
        flow_version_prefix = inputDict.get('flow_version_prefix', 'base').lower()
        version_prefix = version_prefix + '_' + flow_version_prefix
        # data check
        prefixList = np.array(['base'])
        assert (flow_version_prefix == prefixList).any(), "Please enter a valid flow version prefix\n Prefix assigned = %s must be in List %s" % (flow_version_prefix, prefixList)
    if morphFlag:
        assert 'morph_version_prefix' in inputDict, 'Must have "morph_version_prefix" in your input yaml'
        morph_version_prefix = inputDict.get('morph_version_prefix', 'base')
        version_prefix = version_prefix + '_' + morph_version_prefix
        # data check
        prefixList = np.array(['FIXED', 'MOBILE', 'MOBILE_RESET'])
        assert (morph_version_prefix == prefixList).any(), "Please enter a valid morph version prefix\n Prefix assigned = %s must be in List %s" % (morph_version_prefix, prefixList)

    # ______________________________________________________________________________
    # define version parameters
    # TODO: is there things i can put up here for wave and flow
    if model in ['cms'] and waveFlag is True:
        simFnameBackground = inputDict.get('gridSIM', 'grids/CMS/CMS-Wave-FRF.sim')
        backgroundDepFname = inputDict.get('gridDEP', 'grids/CMS/CMS-Wave-FRF.dep')
        # do versioning stuff here
        if wave_version_prefix in ['base', 'HP']:
            full = False
        cmsio = inputOutput.cmsIO()  # initializing the I/o Script writer
        hotStartFlag = False
        print('TODO: Define hotstart flag from bathy times input to this function ')
    if flowFlag:
        # call cold starts here if time start is in cold start list
        if inputDict['csFlag'] == 1:
            durationRamp = 1  # this is the ramp duration in days
        else:
            durationRamp = 0
        cmsfio = inputOutput.cmsfIO()  # initializing the I/o Script writer
    if morphFlag:
        # do some stuff with morph here
        t = 1
    # _______________________________________________________________________________
    # set times

    try:
        d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
        d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
        date_str = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime
        if inputDict['csFlag'] == 1:
            d1F = d1 - DT.timedelta(hours=int(24*durationRamp))
        else:
            # new plan - if not coldstart, then we need to look up the hotstart file from the previous run and
            # see how far back we need to pull data for to write the boundary condition file
            datePrevious = d1 - DT.timedelta(0, timerun * 3600, 0)
            date_strPrevious = datePrevious.strftime('%Y-%m-%dT%H%M%SZ')
            hotStartFromPath = os.path.join(path_prefix + date_strPrevious, 'ASCII_HotStart')
            # load hot-start files
            cmsfio = inputOutput.cmsfIO()
            cmsfio.read_CMSF_etaDict(hotStartFromPath)
            durationMod = int(cmsfio.eta_dict['time'])
            d1F = d1 - DT.timedelta(hours=durationMod)
    except ValueError:
        assert len(startTime) == 10, 'Your Time does not fit convention, check T/Z and input format'
        d1 = DT.datetime.strptime(startTime, '%Y-%m-%d') + DT.timedelta(TOD / 24., 0, 0)
        d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
        date_str = d1.strftime('%Y-%m-%d')  # used to be endtime
        if inputDict['csFlag'] == 1:
            d1F = d1 - DT.timedelta(hours=int(24*durationRamp))
        else:
            # new plan - if not coldstart, then we need to look up the hotstart file from the previous run and
            # see how far back we need to pull data for to write the boundary condition file
            datePrevious = d1 - DT.timedelta(0, timerun * 3600, 0)
            date_strPrevious = datePrevious.strftime('%Y-%m-%dT%H%M%SZ')
            hotStartFromPath = os.path.join(path_prefix + date_strPrevious, 'ASCII_HotStart')
            # load hot-start files
            cmsfio = inputOutput.cmsfIO()
            cmsfio.read_CMSF_etaDict(hotStartFromPath)
            durationMod = int(cmsfio.eta_dict['time'])
            d1F = d1 - DT.timedelta(hours=durationMod)
        assert int(timerun) >= 24, 'Running Simulations with less than 24 Hours of simulation time require end ' \
                                   'Time format in type: %Y-%m-%dT%H:%M:%SZ'

    # create all time lists
    waveTs = inputDict['wave_time_step']
    flowTs = inputDict['flow_time_step']
    morphTs = inputDict['morph_time_step']
    TS_list = [waveTs, flowTs, morphTs]
    mTS = np.nanmin(np.array(TS_list, dtype=np.float64))
    timeList = [d1F + DT.timedelta(minutes=mTS*x) for x in range(0, int((1440/float(mTS))*(d2-d1F).days + (d2-d1F).seconds/float(60*mTS)))]
    waveTimeList = [d1 + DT.timedelta(minutes=waveTs*x) for x in range(0, int((1440/float(waveTs))*(d2-d1).days + (d2-d1).seconds/float(60*waveTs)))]
    flowTimeList = [d1F + DT.timedelta(minutes=flowTs * x) for x in range(0, int((1440 / float(flowTs)) * (d2 - d1F).days + (d2 - d1F).seconds / float(60 * flowTs)))]
    morphTimeList = [d1 + DT.timedelta(minutes=morphTs * x) for x in range(0, int((1440 / float(morphTs)) * (d2 - d1).days + (d2 - d1).seconds / float(60 * morphTs)))]
    # TODO why do i need time steps for every model?
    # __________________Make Diretories_____________________________________________
    #
    if not os.path.exists(os.path.join(path_prefix, date_str)):  # if it doesn't exist
        os.makedirs(os.path.join(path_prefix, date_str))  # make the directory
    if not os.path.exists(os.path.join(path_prefix, date_str, 'figures')):
        os.makedirs(os.path.join(path_prefix, date_str, "figures"))

    print("Model Time Start : %s  Model Time End:  %s" % (d1, d2))
    print(u"OPERATIONAL files will be place in {0} folder".format(os.path.join(path_prefix, date_str)))
    ###############################3###################################3###################################3############
    ###################################3###################################3###################################3########
    ### ____________ Get bathy grid from server ________________
    ###################################3###################################3###################################3########
    ###################################3###################################3###################################3########
    gdTB = getDataTestBed(d1, d2)
    prepdata = STPD.PrepDataTools()
    go = getObs(d1F, d2, THREDDS=inputDict['THREDDS'])  # initialize get observation
    #bathy = gdTB.getGridCMS(method='historical')
    bathy = gdTB.getBathyIntegratedTransect(method=1)
    # ______________________________________________________________________________
    # begin model data gathering
    ## _____________WINDS______________________
    print('_________________\nGetting Wind Data')
    try:
        rawwind = go.getWind(gaugenumber=0)
        # average and rotate winds
        windpacket = prepdata.prep_wind(rawwind, timeList)
        # wind height correction
        print('number of wind records %d with %d interpolated points' % (np.size(windpacket['time']), sum(windpacket['flag'])))
    except (RuntimeError, TypeError):
        windpacket = None
        print(' NO WIND ON RECORD')

    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    try:
        # get water level data
        rawWL = go.getWL()
        # average WL
        WLpacket = prepdata.prep_WL(rawWL, timeList)
        print('number of WL records %d, with %d interpolated points' % (np.size(WLpacket['time']), sum(WLpacket['flag'])))
    except (RuntimeError, TypeError):
        WLpacket = None

    # write the files I need for the different parts of CMS i am running
    if waveFlag:
        ## _____________WAVES____________________________
        rawspec = go.getWaveSpec(gaugenumber=0)
        assert rawspec is not None, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (d1, d2)

        # rotate and lower resolution of directional wave spectra
        wavepacket = prepdata.prep_spec(rawspec, wave_version_prefix, datestr=date_str, plot=pFlag, full=full, outputPath=path_prefix, CMSinterp=50) # 50 freq bands are max for model
        print("number of wave records %d with %d interpolated points" % (np.shape(wavepacket['spec2d'])[0], wavepacket['flag'].sum()))

        # modify packets for different time-steps!
        windpacketW, WLpacketW, wavepacketW = prepdata.mod_packets(waveTimeList, windpacket, WLpacket, wavepacket=wavepacket)

        bathyWaves = prepdata.prep_CMSbathy(bathy, simFnameBackground, backgroundGrid=backgroundDepFname)
        ### ___________ Create observation locations ________________ # these are cell i/j locations
        gaugeLocs = [[1, 25],     # Waverider 26m
                     [49, 150],   # waverider 17m
                     [212, 183],  # awac 11m
                     [251, 183],  # 8m
                     [282, 183],  # 6m
                     [303, 183],  # 4.5m
                     [313, 183],  # 3.5mS
                     [323, 183],  # xp200m
                     [328, 183],  # xp150m
                     [330, 183]]  # xp125m

        ## begin output file name creation
        stdFname = os.path.join(path_prefix + date_str, date_str + '.std')
        simFnameOut = os.path.join(path_prefix + date_str,  date_str +'.sim')
        specFname = os.path.join(path_prefix + date_str, date_str +'.eng')
        bathyFname = os.path.join(path_prefix + date_str, date_str + '.dep')

        # origin = cmsio.ReadCMS_sim(simFnameBackground)
        gridOrigin = (bathyWaves['x0'], bathyWaves['y0'])
        # write files
        cmsio.writeCMS_std(fname=stdFname, gaugeLocs=gaugeLocs)
        cmsio.writeCMS_sim(simFnameOut, date_str, gridOrigin)
        cmsio.writeCMS_spec(specFname, wavePacket=wavepacketW, wlPacket=WLpacketW, windPacket=windpacketW)
        cmsio.writeCMS_dep(bathyFname, depPacket=bathyWaves)
        stio = inputOutput.stwaveIO('')
        stio.write_flags(date_str, path_prefix, wavepacketW, windpacketW, WLpacketW, curpacket=None)
        cmsio.clearAllSimFiles(path_prefix+date_str) # remove old output files so they're not appended (cms default)

    if flowFlag:
        # check to be sure the .tel file is in the inputYaml
        assert 'gridTEL' in inputDict.keys(), 'Error: to run CMS-Flow a .tel file must be specified.'

        # modify packets for different time-steps!
        windpacketF, WLpacketF, wavepacketF = prepdata.mod_packets(flowTimeList, windpacket, WLpacket)

        #################### PREP DATA ################################################################################
        # now we need to write the .xys files for these... - note, .bid file is always the same, so no need to write.
        cmCards, windDirDict, windVelDict, wlDict = prepdata.prep_dataCMSF(windpacketF, WLpacketF, bathy, inputDict, hotStartFlag)
        HSfname = 'ASCII_HotStart'
        cmsfio.clearAllSimFiles(path_prefix + date_str,
                                hotStart=hotStartFlag)  # clear previous sim files before writing hotstarts
        ###################### end Prep Data for flow ##################################################################


        ############# write CMS Flow files #############################################################################
        ncgYaml = 'yaml_files/BATHY/CMSFtel0_global.yml'
        ncvYaml = 'yaml_files/BATHY/CMSFtel0_var.yml'
        makenc.makenc_CMSFtel(ofname=os.path.join(path_prefix + date_str, date_str + '.nc'), dataDict=cmsfio.telnc_dict,
                              globalYaml=ncgYaml, varYaml=ncvYaml)
        cmsfio.write_CMSF_tel(path=path_prefix + date_str, telDict=cmsfio.telnc_dict)
        cmsfio.write_CMSF_cmCards(path=path_prefix + date_str, inputDict=cmCards)
        cmsfio.write_CMSF_xys(path=path_prefix + date_str, xysDict=windDirDict)
        cmsfio.write_CMSF_xys(path=path_prefix + date_str, xysDict=windVelDict)
        cmsfio.write_CMSF_xys(path=path_prefix + date_str, xysDict=wlDict)

        ########### came from above function  ###########################
        # copy over the executable
        shutil.copy2(codeDir + '%s' %inputDict['flowExecutable'], datadir)
        # copy over the .bid file
        tempDir = os.getcwd().split('cmtb')
        shutil.copy2(os.path.join(tempDir[0], 'cmtb/grids/CMS/CMS-Flow-FRF.bid',), datadir)
        # rename this file
        os.rename('CMS-Flow-FRF.bid', ''.join(time.split(':')) + '.bid')
    if morphFlag:
        raise NotImplementedError

def CMSanalyze(startTime, inputDict):
    """
    This runs the analyze script for cms
        Author: Spicer Bak
    Association: USACE CHL Field Research Facility
    Project:  Coastal Model Test Bed
    This Function is the master call for the  data preperation for
    the Coastal Model Test Bed (CMTB).  It is designed to pull from
    GetData and utilize prep_datalib for development of the FRF CMTB

    :param time:
    :param inputDict: this is an input dictionary that was generated with the keys from the project input yaml file
    :return:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory
    """
    # ___________________define Global Variables___________________________________
    if 'pFlag' in inputDict:
        pFlag = inputDict['pFlag']
    else:
        pFlag = True  # will plot true by default

    # version prefixes!
    wave_version_prefix = inputDict['wave_version_prefix']

    path_prefix = inputDict['path_prefix'] #  + "/%s/" %wave_version_prefix   # 'data/CMS/%s/' % wave_version_prefix  # for organizing data
    TOD = 0  # Time of day simulation to start
    simulationDuration = inputDict['duration']
    if 'netCDFdir' in inputDict:
        Thredds_Base = inputDict['netCDFdir']
    else:
        whoami = check_output('whoami', shell=True)[:-1]
        Thredds_Base = '/home/%s/thredds_data/' % whoami

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = path_prefix + datestring + '/'
    # ____________________________________________________________________________
    if wave_version_prefix == 'HP':
        full = False
    elif wave_version_prefix == 'UNTUNED':
        full = False
    else:
        pass

    # _____________________________________________________________________________

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    print('Analyzing simulation')
    go = getDataFRF.getObs(d1, d2)  # setting up get data instance
    prepdata = STPD.PrepDataTools()  # initializing instance for rotation scheme
    cio = cmsIO()  # =pathbase) looks for model output files in folder to analyze

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    t=DT.datetime.now()
    print('Loading files ')
    cio.ReadCMS_ALL(fpath)  # load all files
    stat_packet = cio.stat_packet  # unpack dictionaries from class instance
    obse_packet = cio.obse_Packet
    dep_pack = cio.dep_Packet
    dep_pack['bathy'] = np.expand_dims(dep_pack['bathy'], axis=0)
    # convert dep_pack to proper dep pack with keys
    wave_pack = cio.wave_Packet
    print('Loaded files in %s' % (DT.datetime.now() - t))
    # correct model outout angles from STWAVE(+CCW) to Geospatial (+CW)
    stat_packet['WaveDm'] = testbedutils.anglesLib.STWangle2geo(stat_packet['WaveDm'])
    # correct angles
    stat_packet['WaveDm'] = testbedutils.anglesLib.angle_correct(stat_packet['WaveDm'])

    obse_packet['ncSpec'] = np.ones(
        (obse_packet['spec'].shape[0], obse_packet['spec'].shape[1], obse_packet['spec'].shape[2], 72)) * 1e-6
    # interp = np.ones((obse_packet['spec'].shape[0], obse_packet['spec'].shape[1], wavefreqbin.shape[0],
    #                   obse_packet['spec'].shape[3])) * 1e-6  ### TO DO marked for removal
    for station in range(0, np.size(obse_packet['spec'], axis=1)):
        # rotate the spectra back to true north
        obse_packet['ncSpec'][:, station, :, :], obse_packet['ncDirs'] = prepdata.grid2geo_spec_rotate(
                obse_packet['directions'],  obse_packet['spec'][:, station, :, :])   #interp[:, station, :, :]) - this was with interp
        # now converting m^2/Hz/radians back to m^2/Hz/degree
        # note that units of degrees are on the denominator which requires a deg2rad conversion instead of rad2deg
        obse_packet['ncSpec'][:, station, :, :] = np.deg2rad(obse_packet['ncSpec'][:, station, :, :])
    obse_packet['modelfreqbin'] = obse_packet['wavefreqbin']
    obse_packet['wavefreqbin'] = obse_packet['wavefreqbin'] # wavefreqbin  # making sure output frequency bins now match the freq that were interped to

    ######################################################################################################################
    ######################################################################################################################
    ##################################  Spatial Data HERE     ############################################################
    ######################################################################################################################
    ######################################################################################################################
    tempClass = prepdata.prepDataLib.PrepDataTools()
    gridPack = tempClass.makeCMSgridNodes(float(cio.sim_Packet[0]), float(cio.sim_Packet[1]),
                                                     float(cio.sim_Packet[2]), dep_pack['dx'], dep_pack['dy'],
                                                     dep_pack['bathy'])# dims [t, x, y]
    # ################################
    #        Make NETCDF files       #
    # ################################
    # STio = stwaveIO()
    if np.median(gridPack['elevation']) < 0:
        gridPack['elevation'] = -gridPack['elevation']

    fldrArch = 'waveModels/CMS/%s/' % wave_version_prefix
    spatial = {'time': nc.date2num(wave_pack['time'], units='seconds since 1970-01-01 00:00:00'),
               'station_name': 'Regional Simulation Field Data',
               'waveHs': np.transpose(wave_pack['waveHs'], (0,2,1)),  # put into dimensions [t, y, x]
               'waveTm': np.transpose(np.ones_like(wave_pack['waveHs']) * -999, (0,2,1)),
               'waveDm': np.transpose(wave_pack['waveDm'], (0,2,1)), # put into dimensions [t, y, x]
               'waveTp': np.transpose(wave_pack['waveTp'], (0,2,1)), # put into dimensions [t, y, x]
               'bathymetry': np.transpose(gridPack['elevation'], (0,2,1)), # put into dimensions [t, y, x]
               'latitude': gridPack['latitude'], # put into dimensions [t, y, x] - NOT WRITTEN TO FILE
               'longitude': gridPack['longitude'], # put into dimensions [t, y, x] - NOT WRITTEN TO FILE
               'xFRF': gridPack['xFRF'], # put into dimensions [t, y, x]
               'yFRF': gridPack['yFRF'], # put into dimensions [t, y, x]
               ######################
               'DX': dep_pack['dx'],
               'DX': dep_pack['dy'],
               'NI': dep_pack['NI'],
               'NJ': dep_pack['NJ'],
               'grid_azimuth': gridPack['azimuth']
               }

    TdsFldrBase = os.path.join(Thredds_Base, fldrArch, 'Field')
    fieldOfname = TdsFldrBase + '/Field%s.nc' % datestring
    if not os.path.exists(TdsFldrBase):
        os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    if not os.path.exists(TdsFldrBase + '/Field.ncml'):
        STio = stwaveIO('')
        STio.makencml(TdsFldrBase + '/Field.ncml')  # remake the ncml if its not there
    # make file name strings
    flagfname = fpath + 'Flags%s.out.txt' % datestring  # startTime # the name of flag file
    fieldYaml = 'yaml_files/%sField_globalmeta.yml' % (fldrArch)  # field
    varYaml = 'yaml_files/%sField_var.yml' % (fldrArch)
    assert os.path.isfile(fieldYaml), 'NetCDF yaml files are not created'  # make sure yaml file is in place
    makenc.makenc_field(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)
    ###################################################################################################################
    ###############################   Plotting  Below   ###############################################################
    ###################################################################################################################
    dep_pack['bathy'] = np.transpose(dep_pack['bathy'], (0,2,1))  # dims [t, y, x]
    plotParams = [('waveHs', 'm'), ('bathymetry', 'NAVD88 $[m]$'), ('waveTp', 's'), ('waveDm', 'degTn')]
    if pFlag == True:
        for param in plotParams:
            print('    plotting %s...' %param[0])
            spatialPlotPack = {'title': 'Regional Grid: %s' % param[0],
                               'xlabel': 'Longshore distance [m]',
                               'ylabel': 'Cross-shore distance [m]',
                               'field': spatial[param[0]],
                               'xcoord': spatial['xFRF'],
                               'ycoord': spatial['yFRF'],
                               'cblabel': '%s-%s' % (param[0], param[1]),
                               'time': nc.num2date(spatial['time'], 'seconds since 1970-01-01')}
            fnameSuffix = '/figures/CMTB_CMS_%s_%s' % (wave_version_prefix, param[0])
            oP.plotSpatialFieldData(dep_pack, spatialPlotPack, fnameSuffix, fpath, nested=0)
            # now make a gif for each one, then delete pictures
            fList = sorted(glob.glob(fpath + '/figures/*%s*.png' % param[0]))
            sb.makegif(fList, fpath + '/figures/CMTB_%s_%s_%s.gif' % (wave_version_prefix, param[0], datestring))
            [os.remove(ff) for ff in fList]

    ######################################################################################################################
    ######################################################################################################################
    ##################################  Wave Station Files HERE (loop) ###################################################
    ######################################################################################################################
    ######################################################################################################################

    # this is a list of file names to be made with station data from the parent simulation
    stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac_4.5m', 'adop-3.5m',
                   'xp200m', 'xp150m', 'xp125m']
    for gg, station in enumerate(stationList):

        stationName = 'CMTB-waveModels_CMS_%s_%s' % (wave_version_prefix, station)  # xp 125

        # this needs to be the same order as the run script
        stat_yaml_fname = 'yaml_files/' + fldrArch + 'Station_var.yml'
        globalyaml_fname = 'yaml_files/' + fldrArch + 'Station_globalmeta.yml'

        # getting lat lon, easting northing idx's
        Idx_i = len(gridPack['i']) - np.argwhere(gridPack['i'] == stat_packet['iStation'][
            gg]).squeeze() - 1  # to invert the coordinates from offshore 0 to onshore 0

        Idx_j = np.argwhere(gridPack['j'] == stat_packet['jStation'][gg]).squeeze()
        w = go.getWaveSpec(station)
        # print('   Comparison location taken from thredds, check positioning ')
        stat_data = {'time': nc.date2num(stat_packet['time'][:], units='seconds since 1970-01-01 00:00:00'),
                     'waveHs': stat_packet['waveHs'][:, gg],
                     'waveTm': np.ones_like(stat_packet['waveHs'][:, gg]) * -999,
                     # this isn't output by model, but put in fills to stay consitant
                     'waveDm': stat_packet['WaveDm'][:, gg],
                     'waveTp': stat_packet['Tp'][:, gg],
                     'waterLevel': stat_packet['waterLevel'][:, gg],
                     'swellHs': stat_packet['swellHs'][:, gg],
                     'swellTp': stat_packet['swellTp'][:, gg],
                     'swellDm': stat_packet['swellDm'][:, gg],
                     'seaHs': stat_packet['seaHs'][:, gg],
                     'seaTp': stat_packet['seaTp'][:, gg],
                     'seaDm': stat_packet['seaDm'][:, gg],

                     'station_name': stationName,
                     'directionalWaveEnergyDensity': obse_packet['ncSpec'][:, gg, :, :],
                     'waveDirectionBins': obse_packet['ncDirs'],
                     'waveFrequency': obse_packet['wavefreqbin'],
                     ###############################
                     'DX': dep_pack['dx'],
                     'DY': dep_pack['dy'],
                     'NI': dep_pack['NI'],
                     'NJ': dep_pack['NJ'],
                     'grid_azimuth': gridPack['azimuth']}
        try:
            stat_data['Latitude'] = w['latitude']
            stat_data['Longitude'] = w['longitude']
        except KeyError:  # this should be rectified
            stat_data['Latitude'] = w['lat']
            stat_data['Longitude'] = w['lon']
        # Name files and make sure server directory has place for files to go
        print('making netCDF for model output at %s ' % station)
        TdsFldrBase = os.path.join(Thredds_Base, fldrArch, station)
        outFileName = TdsFldrBase + '/'+stationName + '_%s.nc' % datestring
        if not os.path.exists(TdsFldrBase):
            os.makedirs(TdsFldrBase)  # make the directory for the file/ncml to go into
        if not os.path.exists(TdsFldrBase + '/' + stationName + '.ncml'):
            STio = stwaveIO('')
            STio.makencml(TdsFldrBase + '/' + stationName + '.ncml')
        # make netCDF
        makenc.makenc_Station(stat_data, globalyaml_fname=globalyaml_fname, flagfname=flagfname,
                              ofname=outFileName, stat_yaml_fname=stat_yaml_fname)

        print("netCDF file's created for station: %s " % station)
        ###################################################################################################################
        ###############################   Plotting  Below   ###############################################################
        ###################################################################################################################


        if pFlag == True and 'time' in w:
            if full == False:
                w['dWED'], w['wavedirbin'] = prepdata.HPchop_spec(w['dWED'], w['wavedirbin'], angadj=70)
            obsStats = sbwave.waveStat(w['dWED'], w['wavefreqbin'], w['wavedirbin'])

            modStats = sbwave.waveStat(obse_packet['ncSpec'][:, gg, :, :], obse_packet['wavefreqbin'],
                                       obse_packet['ncDirs'])  # compute model stats here

            time, obsi, modi = sb.timeMatch(nc.date2num(w['time'], 'seconds since 1970-01-01'),
                                            np.arange(w['time'].shape[0]),
                                            nc.date2num(stat_packet['time'][:], 'seconds since 1970-01-01'),
                                            np.arange(len(stat_packet['time'])))  # time match

            for param in modStats:  # loop through each bulk statistic
                print('    plotting %s: %s' %(station, param))
                if param in ['Tp', 'Tm10']:
                    units = 's'
                    title = '%s period' % param
                elif param in ['Hm0']:
                    units = 'm'
                    title = 'Wave Height %s ' % param
                elif param in ['Dm',  'Dp']:
                    units = 'degrees'
                    title = 'Direction %s' % param
                # elif param in ['sprdF', 'sprdD']:
                #     units = ''
                #     title = 'Spread %s ' % param

                # now run plots
                if param in ['meta','Tm', 'sprdF', 'VecAvgMeanDir','Tave', 'Dmp','sprdD']:
                    pass
                else:
                    p_dict = {'time': nc.num2date(time, 'seconds since 1970-01-01'),
                              'obs': obsStats[param][obsi.astype(int)],
                              'model': modStats[param][modi.astype(int)],
                              'var_name': param,
                              'units': units,  # ) -> this will be put inside a tex math environment!!!!
                              'p_title': title}
                    ofname = fpath + 'figures/Station_%s_%s_%s' % (station, param, datestring)

                    # need to check here if your data is chock full of nans?
                    # this happened for the xp200m Wave Height Hm0 data for the 10-01-2015 CMS run.
                    # I don't know how big of a problem this is... but they were all nan, like every single point...
                    # just going to nanmin and nanmax on the axis labels is not going to fix that...
                    if sum(np.isnan(p_dict['obs'])) > len(p_dict['obs']) - 5:
                        pass
                    elif sum(np.isnan(p_dict['model'])) > len(p_dict['model']) - 5:
                        pass
                    else:
                        obs_V_mod_TS(ofname, p_dict, logo_path='ArchiveFolder/CHL_logo.png')

                    if station == 'waverider-26m' and param == 'Hm0':
                        # this is a fail safe to abort run if the boundary conditions don't
                        # meet qualtity standards below
                        bias = 0.1  # bias has to be within 10 centimeters
                        RMSE = 0.1  # RMSE has to be within 10 centimeters
                        stats = sb.statsBryant(p_dict['obs'], p_dict['model'])
                        try:
                            assert stats['RMSE'] < RMSE, 'RMSE test on spectral boundary energy failed'
                            assert np.abs(stats['bias']) < bias, 'bias test on spectral boundary energy failed'
                        except:
                            print('!!!!!!!!!!FAILED BOUNDARY!!!!!!!!')
                            print('deleting data from thredds!')
                            os.remove(fieldOfname)
                            os.remove(outFileName)
                            raise RuntimeError('The Model Is not validating its offshore boundary condition')

def CMSFanalyze(startTime, inputDict):
    """
    This runs the analyze script for cmsflow
        Author: David Young
    Association: USACE CHL Field Research Facility
    Project:  Coastal Model Test Bed
    This Function is the master call for the  data preperation for
    the Coastal Model Test Bed (CMTB).  It is designed to pull from
    GetData and utilize prep_datalib for development of the FRF CMTB

    :param startTime: this is the start time of this model run
    :param inputDict: this is an input dictionary that was generated with the keys from the project input yaml file
    :return:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory
    """

    # ___________________define Global Variables___________________________________
    if 'pFlag' in inputDict:
        pFlag = inputDict['pFlag']
    else:
        pFlag = True  # will plot true by default

    # version prefixes!
    flow_version_prefix = inputDict['flow_version_prefix']

    path_prefix = inputDict['path_prefix']  # + "/%s/" %wave_version_prefix   # 'data/CMS/%s/' % wave_version_prefix  # for organizing data
    TOD = 0  # Time of day simulation to start
    simulationDuration = inputDict['duration']
    if 'netCDFdir' in inputDict:
        Thredds_Base = inputDict['netCDFdir']
    else:
        whoami = check_output('whoami', shell=True)[:-1]
        Thredds_Base = '/home/%s/thredds_data/' % whoami

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = path_prefix + datestring + '/'

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    print('Analyzing simulation')
    go = getDataFRF.getObs(d1, d2)  # setting up get data instance
    prepdata = STPD.PrepDataTools()  # initializing instance for rotation scheme
    cio = cmsfIO()  # looks for model output files in folder to analyze

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    t = DT.datetime.now()
    print('Loading files ')

    # load the files now
    cio.read_CMSF_all(path=fpath)
    # i think the only change i have to make is to convert the velocities into the same coord system as the gages?

    """
    # rotate my velocity vectors CLOCKWISE by the grid azimuth!
    print('Rotating velocity vectors - this part is slow...')
    test_fun = lambda x: testbedutils.anglesLib.vectorRotation(x, theta=cio.telnc_dict['azimuth'])
    [rows, cols] = np.shape(cio.vel_dict['vx'])
    newVx = np.zeros(np.shape(cio.vel_dict['vx']))
    newVy = np.zeros(np.shape(cio.vel_dict['vx']))
    # this right here is incredibly slow...
    for ii in range(0, rows):
        for jj in range(0, cols):
            vel = [cio.vel_dict['vx'][ii, jj], cio.vel_dict['vy'][ii, jj]]
            newvel = test_fun(vel)
            newVx[ii, jj] = newvel[0]
            newVy[ii, jj] = newvel[1]
    """
    # debugging only!
    newVx = cio.vel_dict['vx']
    newVy = cio.vel_dict['vy']

    cmsfWrite = {}

    # IMPORTANT NEW INFORMATION!!
    # the output files DO NOT have all nodes, ONLY the nodes that are NOT -999
    # so I guess we save the not -999 nodes only?
    # it LOOKS LIKE the .xy file has the list of all the computed nodes in it, so we will use that to cross-reference?
    # is the order of the solution nodes the same as the .xy nodes? yes - according to CMS users manual DLY 06/18/2018.

    # OTHER NEW INFORMATION!!!
    # the nodes in the .xy file appear to be the non -999 cellID's in the .tel file in REVERSE!  so higher cellID's
    # are on top - see testing in DLY_telFile that I did on 6/19/2018.  THE "nodeIndex" in the xy_dict is NOT THE SAME
    # as the cellID's in the tel file!!!!!!!!!!!!!!

    # velocity info
    cmsfWrite['aveE'] = newVx.copy()
    cmsfWrite['aveN'] = newVy.copy()
    del newVx
    del newVy
    # water level info
    cmsfWrite['waterLevel'] = cio.eta_dict['wl']  # -999 is masked
    # general parameters
    cmsfWrite['durationRamp'] = cio.cmcards_dict['durationRamp']

    # time info
    timeunits = 'seconds since 1970-01-01 00:00:00'
    if cmsfWrite['durationRamp'] > 0:
        d1F = d1 - DT.timedelta(hours=int(24*cmsfWrite['durationRamp']))
    else:
        d1F = d1

    # tstep = inputDict['flow_time_step']  NO NO NO NO!  OUTPUTS HOURLY REGARDLESS OF TIMESTEP!
    cmsfWrite['time'] = nc.date2num(np.array([d1F + DT.timedelta(0, x * 3600, 0) for x in cio.vel_dict['time']]), timeunits)

    # okay, I have a sneaking suspicion that sb will want to go the masked array route with the whole .tel file
    # worth of nodes?  maybe?
    cmsfWriteN = prepdata.modCMSFsolnDict(cmsfWrite, cio.telnc_dict, inputDict['duration'])
    del cmsfWrite
    cmsfWrite = cmsfWriteN
    del cmsfWriteN

    # now hand this to makenc_CMSFrun?
    print('Writing simulation netCDF files.')
    ofname = datestring + '.nc'  # you may need to come back to this to check on it.
    TdsFldrBase = os.path.join(Thredds_Base, 'CMSF')
    if not os.path.exists(TdsFldrBase):
        os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    ncgYaml = '/home/david/PycharmProjects/cmtb/yaml_files/CMSF/CMSFrun_global.yml'
    ncvYaml = '/home/david/PycharmProjects/cmtb/yaml_files/CMSF/CMSFrun_var.yml'
    makenc.makenc_CMSFrun(os.path.join(TdsFldrBase, ofname), cmsfWrite, ncgYaml, ncvYaml)

    # now make the plots off the cmsfWrite dictionary?
    # create velocity magnitude dataset
    cmsfWrite['vMag'] = np.sqrt(np.power(cmsfWrite['aveE'], 2) + np.power(cmsfWrite['aveN'], 2))

    # i think that cmsfWrite gets modified during makenc_CMSFrun?  so I need to put depth back into the
    # keys if it gets converted to elevation earlier?
    if 'depth' not in cmsfWrite.keys():
        cmsfWrite['depth'] = -1*cmsfWrite['elevation']

    # mask all values where WATER LEVEL?! is -999?!?!?!?
    maskInd = cmsfWrite['waterLevel'] == -999

    newVx = np.ma.masked_where(maskInd, cmsfWrite['aveE'])
    newVy = np.ma.masked_where(maskInd, cmsfWrite['aveN'])
    newWL = np.ma.masked_where(maskInd, cmsfWrite['waterLevel'])
    newvMag = np.ma.masked_where(maskInd, cmsfWrite['vMag'])
    newDepth = np.ma.masked_where(cmsfWrite['depth'] == -999, cmsfWrite['depth'])
    del cmsfWrite['waterLevel']
    del cmsfWrite['aveE']
    del cmsfWrite['aveN']
    del cmsfWrite['vMag']
    del cmsfWrite['depth']
    cmsfWrite['vMag'] = newvMag
    cmsfWrite['waterLevel'] = newWL
    cmsfWrite['aveE'] = newVx
    cmsfWrite['aveN'] = newVy
    cmsfWrite['depth'] = newDepth

    # tack on the xFRF and yFRF values since I DONT have them in the solution netcdf files anymore
    cmsfWrite['xFRF'] = cio.telnc_dict['xFRF']
    cmsfWrite['yFRF'] = cio.telnc_dict['yFRF']

    ###################################################################################################################
    ###############################   Plotting  Below   ###############################################################
    ###################################################################################################################
    # .gif's
    """
    # plotParams = [('WL', 'm', 'waterLevel'), ('depth', 'NAVD88 $[m]$', 'depth'), ('VelMag', 'm/s', 'vMag')]
    plotParams = [('WL', 'm', 'waterLevel'), ('VelMag', 'm/s', 'vMag')]
    if pFlag:
        for param in plotParams:
            print('    plotting %s...' % param[0])
            for ss in range(0, len(cmsfWrite['time'])):

                pDict = {'ptitle': 'Regional Grid: %s' % param[0],
                         'xlabel': 'xFRF [m]',
                         'ylabel': 'yFRF [m]',
                         'x': cmsfWrite['xFRF'],
                         'y': cmsfWrite['yFRF'],
                         'z': cmsfWrite[param[2]][ss, :],
                         'cbarMin': np.nanmin(cmsfWrite[param[2]]) - 0.05,
                         'cbarMax': np.nanmax(cmsfWrite[param[2]]) + 0.05,
                         'cbarColor': 'coolwarm',
                         'xbounds': (-50, 2000),
                         'ybounds': (-1000, 2000),
                         'cbarLabel': param[1],
                         'gaugeLabels': True}
                numStr = "%02d" %ss
                fnameSuffix = 'figures/CMTB_CMSF_%s_%s_%s' % (flow_version_prefix, param[0], numStr)
                noP.plotUnstructBathy(ofname=os.path.join(fpath, fnameSuffix), pDict=pDict)

            # now make a gif for each one, then delete pictures
            fList = sorted(glob.glob(fpath + '/figures/*%s*.png' % param[0]))
            sb.makegif(fList, fpath + '/figures/CMTB_CMSF_%s_%s_%s.gif' % (flow_version_prefix, param[0], datestring))
            [os.remove(ff) for ff in fList]
    """
    # stations
    if pFlag:
        stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m',
                       'xp200m', 'xp150m', 'xp125m']
        exclude = set(string.punctuation)
        for station in stationList:

            # velocity plots first
            if 'awac' in station or 'adop' in station:

                oDict = getPlotData.CMSF_velData(cmsfWrite, station)
                if oDict is None:
                    print('Velocity data missing for %s' %station)
                    pass

                else:
                    if len(oDict['time']) <= 1:
                        pass
                    else:
                        # ave E velocity
                        path = fpath + '/figures/CMTB_CMSF_%s_%s_%s_aveE.png' % (flow_version_prefix, datestring, ''.join(ch for ch in station if ch not in exclude).strip())
                        p_dict = {'time': oDict['time'],
                                  'obs': oDict['aveEobs'],
                                  'model': oDict['aveEmod'],
                                  'var_name': '$\overline{U}$',
                                  'units': 'm/s'}
                        oP.obs_V_mod_TS(path, p_dict)
                        # ave N velocity
                        path = fpath + '/figures/CMTB_CMSF_%s_%s_%s_aveN.png' % (flow_version_prefix, datestring, ''.join(ch for ch in station if ch not in exclude).strip())
                        p_dict = {'time': oDict['time'],
                                  'obs': oDict['aveNobs'],
                                  'model': oDict['aveNmod'],
                                  'var_name': '$\overline{V}$',
                                  'units': 'm/s'}
                        oP.obs_V_mod_TS(path, p_dict)

            # what stations have water level?
            if 'waverider' not in station:

                oDict = getPlotData.CMSF_wlData(cmsfWrite, station)
                if oDict is None:
                    print('Water level data missing for %s' % station)
                    pass

                else:
                    if len(oDict['time']) <= 1:
                        pass
                    else:
                        # water level
                        path = fpath + '/figures/CMTB_CMSF_%s_%s_%s_WL.png' % (flow_version_prefix, datestring, ''.join(ch for ch in station if ch not in exclude).strip())
                        p_dict = {'time': oDict['time'],
                                  'obs': oDict['obsWL'],
                                  'model': oDict['modWL'],
                                  'var_name': '$WL (NAVD88)$',
                                  'units': 'm'}
                        oP.obs_V_mod_TS(path, p_dict)

















