import datetime as DT
import os, glob, shutil, string, makenc
from subprocess import check_output
import netCDF4 as nc
import numpy as np
from getdatatestbed import getDataFRF
from testbedutils import sblib as sb
from testbedutils import waveLib as sbwave
from testbedutils import fileHandling
from testbedutils import anglesLib, gridTools
from prepdata import prepDataLib
from prepdata import inputOutput
from plotting import operationalPlots as oP

def modStartTimes(oldStartTime, modifiedStartTime, inputDict, datestring):
    """
    modifies start time/end time for going into wave setup when running a cold start, puts it back to the way it was if
    running off of hot-start

    Args:
        oldStartTime: start time of original simulation
        modifiedStartTime: datetime object as output from CMSFlow
        inputDict: input dictionary read from yaml steering file
        datestring(str): to save files use this string

    Returns:
        modified datestring and inputDict

    """

    if isinstance(oldStartTime, str):
        oldStartTime = DT.datetime.strptime(oldStartTime, "%Y-%m-%dT%H:%M:%SZ")
    if modifiedStartTime != oldStartTime:             # i have to modify the start times for the wave simulation
        inputDict['org_simulationDuration'] = inputDict['simulationDuration']
        inputDict['simulationDuration'] = np.abs(modifiedStartTime - oldStartTime).total_seconds()/60/60 + inputDict['simulationDuration']
        inputDict['startTime'] = modifiedStartTime.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif 'org_simulationDuration' in inputDict:                # return input dict back to the way it was
        inputDict['simulationDuration'] = inputDict.pop('org_simulationDuration')

    #set file name base for naming conventions
    inputDict['datestring'] = datestring # oldStartTime.strftime("%Y%m%dT%H%M%SZ")

    return modifiedStartTime.strftime("%Y-%m-%dT%H:%M:%SZ"), inputDict

def CMSFsimSetup(startTime, inputDict, **kwargs):
    """This Function is the master call for the preprocessing for unstructured grid model runs
        it is designed to pull from GetData and utilize prep_datalib for development of the FRF CMTB

    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime(str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict(dict): this is a dictionary that is read from the yaml read function
             requires keys ['startTime', 'endTime', 'path_prefix','version_prefix']  probably others (please add)

    Keyword Args:
        allBathyTimes = dictionary with pre-gathered bathytimes, use this for coldstart/hotstart logic
        allWaves = dictionary with pregathered Waves
        allWind = dictionary with pregathered Wind
        allWL = dictionary with pregathered WL
        durationRamp(int): duration to run the ramp if doing a coldstart

    Returns:
        start Date and run duration (to cover in cases in need to go to cold start)

    """
    # begin by setting up input parameters
    if 'org_simulationDuration' in inputDict:
        # reset sim duration, if the previous run was extra long to match the last bathy time (plus run duration)
        simulationDuration = inputDict['org_simulationDuration']
    else:
        simulationDuration = inputDict.get('simulationDuration', 24)
    
    # pFlag = inputDict.get('plotFlag', True)  # no plots generated
    
    model = inputDict['modelSettings'].get('model', 'CMS').lower()
    durationRamp = inputDict['modelSettings'].get('rampDuration', 1)
    bathyTimes = kwargs.get('bathyTimes', None)
    morphFlag = inputDict['morphSettings'].get('morphFlag', False)
    #TODO: check if better way to handle flags at begining of process
    path_prefix = inputDict['path_prefix']
    version_prefix = inputDict['modelSettings'].get('version_prefix', 'base').lower()
    # flow_version_prefix = inputDict['flowSettings'].get('flow_version_prefix', 'base').lower()
    # version_prefix = version_prefix + '_f' + flow_version_prefix
    #
    # data check
    
    # morphFlag = inputDict['morphSettings'].get('morphFlag', False)
    # if morphFlag:
    #     morph_version_prefix = inputDict.get('morph_version_prefix', 'base').lower()
    #     # data check
    #     prefixList = np.array(['BASE'])
    #     assert (morph_version_prefix.upper() == prefixList).any(), "Please enter a valid morph version prefix\n " \
    #                                                               "Prefix assigned = {} must be in list {}".format(
    #                                                               morph_version_prefix, prefixList)
    #     version_prefix = version_prefix + '_m' + morph_version_prefix
    #
    # _______________________________________________________________________________
    # _______________________________________________________________________________
    # set times (based on hot-start logic)
    # _______________________________________________________________________________
    # _______________________________________________________________________________
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(hours=simulationDuration)
    # inputDict['csFlag'] = 0
    prepdata = prepDataLib.PrepDataTools()                                                      # initialize prep data for preparation
    cmsfio = inputOutput.cmsfIO(path=os.path.join(path_prefix, d1.strftime('%Y%m%dT%H%M%SZ')))  # initializing the I/o Script writer
    cmsfio.simulationDuration = simulationDuration
    cmsfio, d1Flow = prepdata.hotStartLogic(d1, cmsfio, bathyTimes, durationRamp)
    # __________________Make Diretories_____________________________________________
    fileHandling.makeCMTBfileStructure(path_prefix, cmsfio.datestring)
    # get data if i don't already have it
    rawwind = kwargs.get('allWind', None)       # go.getWind(gaugenumber=0))
    rawWL = kwargs.get('allWL', None)           # go.getWL())
    rawspec = kwargs.get('allWaves', None)      # go.getWaveSpec(gaugenumber=0))
    # rawWL = go.getWL() if rawWL is None
    timeList, waveTimeList, flowTimeList, morphTimeList = prepdata.createDifferentTimeLists(d1, d2, rawspec, rawWL,
                                                                                            d1Flow=d1Flow)
    ##################################################################################################################
    print
    print('-----> pre-processing flow')
    gdTB = getDataFRF.getDataTestBed(d1, d2)                        # initalize bathy retrival
    if newBathy is True:
        bathy = gdTB.getBathyIntegratedTransect(method=1)
    else:
        print('loadBathy from last simulation')
    ## _____________WINDS average and rotate winds & correct elevation to 10m ____________
    windpacket = prepdata.prep_wind(rawwind, flowTimeList, model=model)
    ## ___________WATER LEVEL average WL__________________
    WLpacket = prepdata.prep_WL(rawWL, flowTimeList)
    # check to be sure the .tel file is in the inputYaml
    assert 'gridTEL' in inputDict['modelSettings'].keys(), 'Error: to run CMS-Flow a .tel file must be specified.'
    # modify packets for different time-steps by interpolating to necessary timestep and removing extra data
    # it's unclear if this step does anything that's not already done above, maybe in future raise error - sb
    windpacketF, WLpacketF, _ = prepdata.mod_packets(flowTimeList, windpacket, WLpacket)

    #################### PREP DATA ################################################################################
    # clear previous sim files before writing hotstarts
    cmsfio.clearAllFlowSimFiles(path=None)  # uses already established with None value
    # now we need to write the .xys files for these... - note, .bid file is always the same, so no need to write.
    cmCards, windDirDict, windVelDict, wlDict, cmsfio = prepdata.prep_dataCMSF(windpacketF, WLpacketF, bathy, inputDict,
                                                                       cmsfio)
    
    ###################### end Prep Data for flow ##################################################################
    if morphFlag:
        # write the dictionary from the yaml write to the cmcards input file
        cmCards['sedTransport'] = inputDict['morphSettings']

    ##______________________________________________________________________________________________________________
    # _____________________ begin file writing for flow  ___________________________________________________________
    ##______________________________________________________________________________________________________________
    ncgYaml = 'yaml_files/flowModels/cmsf/{}/CMSFtel0_global.yml'.format(version_prefix)
    ncvYaml = 'yaml_files/flowModels/cmsf/CMSFtel0_var.yml'
    makenc.makenc_CMSFtel(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring + '_tel.nc'),
                          dataDict=cmsfio.telnc_dict, globalYaml=ncgYaml, varYaml=ncvYaml)
    cmsfio.write_CMSF_tel(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring + '.tel'),
                          telDict=cmsfio.telnc_dict)
    cmsfio.write_CMSF_xys(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring+'_wind_dir.xys'),
                          xysDict=windDirDict)
    cmsfio.write_CMSF_xys(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring+'_wind_vel.xys'),
                          xysDict=windVelDict)
    cmsfio.write_CMSF_xys(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring+'_h_1.xys'),
                          xysDict=wlDict)
    cmsfio.write_CMSF_cmCards(ofname=os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring + '.cmcards'),
                              inputDict=cmCards)

    # copy over the executable
    shutil.copy2(inputDict['modelExecutable'], os.path.join(path_prefix, cmsfio.datestring))
    # copy over the .bid file
    shutil.copy2('grids/CMS/CMS-Flow-FRF.bid', os.path.join(path_prefix, cmsfio.datestring, cmsfio.datestring+'.bid'))

    cmsfio.version_prefix = version_prefix  # write this for loading purposes
    return d1Flow, d2, cmsfio, waveTimeList

def CMSwaveSimSetup(startTime, inputDict, **kwargs):
    """This Function is the master call for the preprocessing for unstructured CMS grid model runs
        it is designed to pull from GetData and utilize prepdatalib for development of the FRF CMTB

    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime(str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict(dict): this is a dictionary that is read from the yaml read function
             requires keys ['startTime', 'endTime', 'path_prefix',]  probably others (please add)

    Keyword Args:
        allBathyTimes = dictionary with pre-gathered bathytimes, use this for coldstart/hotstart logic
        allWaves = dictionary with pregathered Waves
        allWind = dictionary with pregathered Wind
        allWL = dictionary with pregathered WL
        flowFlag(bool): if also running flow model too

    """
    timerun = inputDict.get('simulationDuration', 24)
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    datestring = inputDict.get('datestring', d1.strftime('%Y%m%dT%H%M%SZ'))
    
    # begin by setting up input parameters
    pFlag = inputDict.get('plotFlag', True)
    path_prefix = inputDict['path_prefix']

    model = inputDict['modelSettings'].get('model', 'CMS').lower()
    simFnameBackground = inputDict['modelSettings'].get('gridSIM', 'grids/CMS/CMS-Wave-FRF.sim')
    backgroundDepFname = inputDict['modelSettings'].get('gridDEP', 'grids/CMS/CMS-Wave-FRF.dep')

    #TODO: check if better way to handle flags at begining of process
    bathyTimes = kwargs.get('bathyTimes', None)
    flowFlag = kwargs.get('flowFlag', False)                                                    # default Run Waves only
    print('TODO: change wave bathy************************************************************************************')
    # define version parameters
    version_prefix = inputDict.get('version_prefix', 'base').lower()
    # wave_version_prefix = version_prefix.split('_')[0]  # wave prefix is always first
    
    # TODO: is there things i can put up here for wave and flow from above??

    # __________________Make Diretories_____________________________________________
    fileHandling.makeCMTBfileStructure(path_prefix, datestring)
    # version_prefix = fileHandling.checkVersionPrefix(model, inputDict)
    #################################################################################################################
    #################################################################################################################
    ### ____________ Get prep data (get it if needed)  ________________
    #################################################################################################################
    #################################################################################################################
    print('------>  pre-processing wave')
    cmsio = inputOutput.cmsIO()                                                     # initializing the I/o Script writer
    prepdata = prepDataLib.PrepDataTools()                                         # intialize prep data for preparation
    
    # go = getDataFRF.getObs(d1, d2)  # initialize get observation incase i need it
    # get data if i don't already have it
    rawwind = kwargs.get('allWind',None)    # go.getWind(gaugenumber=0)) this default gets called for some reason
    rawWL = kwargs.get('allWL', None)       #  go.getWL())
    rawspec = kwargs.get('allWaves', None)  #go.getWaveSpec(gaugenumber=0))
    ############################
    gdTB = getDataFRF.getDataTestBed(d1, d2)                                                  # initalize bathy retrival
    bathy = gdTB.getBathyIntegratedTransect(method=1)
    full = False                                                           # CMS wave doesn't operate in full plane mode
    # create time list for wave model time step, grab from input dict (if was created from flow, need 1 extra value to
    # bound start and finish of flow runs (will have to figure out a way to throw last value out)
    waveTs = np.median(np.diff(rawspec['epochtime']))   # time step in minutes of observations
    waveTimeList = inputDict.get('waveTimeList', [d1 + DT.timedelta(seconds=waveTs * x) for x in
                        range(int((d2 - d1).total_seconds() / float( waveTs)))])

    # __________________________________________________________________________________________________________________
    ## _____________WINDS______________________
    # average and rotate winds & correct elevation to 10m
    windpacket = prepdata.prep_wind(rawwind, waveTimeList, model=model)

    ## ___________WATER LEVEL__________________
    # average WL
    WLpacket = prepdata.prep_WL(rawWL, waveTimeList)

    ## _____________WAVES____________________________
    # rotate and lower resolution of directional wave spectra -- 50 freq bands are max for cms model
    wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=datestring, plot=pFlag, full=full,
                                    outputPath=path_prefix, CMSinterp=50, model=model, waveTimeList=waveTimeList)
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
    ##______________________________________________________________________________________________________________
    # _____________________ begin file writing for waves ___________________________________________________________
    ##______________________________________________________________________________________________________________
    ## begin output file name creation
    stdFname = os.path.join(path_prefix, datestring, datestring + '.std')
    simFnameOut = os.path.join(path_prefix, datestring,  datestring +'.sim')
    specFname = os.path.join(path_prefix, datestring, datestring +'.eng')
    bathyFname = os.path.join(path_prefix, datestring, datestring + '.dep')

    # modify packets for different time-steps!

    windpacketW, WLpacketW, wavepacketW = prepdata.mod_packets(waveTimeList, windpacket, WLpacket, wavepacket=wavepacket)
    # write files
    cmsio.writeCMS_std(fname=stdFname, gaugeLocs=gaugeLocs)
    cmsio.writeCMS_sim(simFnameOut, datestring, origin=(bathyWaves['x0'], bathyWaves['y0']))
    cmsio.writeCMS_spec(specFname, wavePacket=wavepacketW, wlPacket=WLpacketW, windPacket=windpacketW)
    cmsio.writeCMS_dep(bathyFname, depPacket=bathyWaves)
    inputOutput.write_flags(datestring, path_prefix, wavepacketW, windpacketW, WLpacketW, curpacket=None)
    cmsio.clearAllWaveSimFiles(os.path.join(path_prefix, datestring), flowFlag = flowFlag) # remove old output files so they're not appended (cms default)

def CMSanalyze(startTime, inputDict):
    """This runs the analyze script for cms model.

    This Function is the master call for the  data preperation for
    the Coastal Model Test Bed (CMTB).  It is designed to pull from
    GetData and utilize prep_datalib for development of the FRF CMTB
    
    Args:
         time:
         inputDict: this is an input dictionary that was generated with the keys from the project input yaml file
    
    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory
        
    """
    # ___________________define Global Variables___________________________________
    pFlag = inputDict.get('plotFlag', True)      # version prefixes!
    wave_version_prefix = inputDict.get('wave_version_prefix', 'base')
    path_prefix = inputDict['path_prefix']
    simulationDuration = inputDict.get('simulationDuration', 24)
    Thredds_Base = inputDict.get('netCDFdir', '/home/%s/thredds_data/' % check_output('whoami', shell=True)[:-1])
    model = inputDict.get('model', 'cms').lower()
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    # use start time from input Dict to define this as it stays static in the case of coldstarts (for coupled sims), while d1 does not
    datestring = inputDict.get('datestring', inputDict['startTime'].replace(':','').replace('-',''))  # a string for file names
    fpath = os.path.join(path_prefix, datestring)
    # ____________________________________________________________________________
    full=False  # CMS wave doesn't have full plane capability
    # _____________________________________________________________________________
    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    print('Analyzing simulation')
    go = getDataFRF.getObs(d1, d2)  # setting up get data instance
    prepdata = prepDataLib.PrepDataTools()                            # intialize prep data for preparation
    cio = inputOutput.cmsIO()  # looks for model output files in folder to analyze

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    t=DT.datetime.now()
    print('Loading files ')
    cio.ReadCMS_ALL(fpath, CMSF=True)  # load all files
    stat_packet = cio.stat_packet  # unpack dictionaries from class instance
    obse_packet = cio.obse_Packet
    dep_pack = cio.dep_Packet
    dep_pack['bathy'] = np.expand_dims(dep_pack['bathy'], axis=0)
    # convert dep_pack to proper dep pack with keys
    wave_pack = cio.wave_Packet
    print('Loaded files in %s' % (DT.datetime.now() - t))
    # correct model outout angles from STWAVE(+CCW) to Geospatial (+CW)
    stat_packet['WaveDm'] = anglesLib.STWangle2geo(stat_packet['WaveDm'])
    # correct angles
    stat_packet['WaveDm'] = anglesLib.angle_correct(stat_packet['WaveDm'])

    obse_packet['ncSpec'] = np.ones(
        (obse_packet['spec'].shape[0], obse_packet['spec'].shape[1], obse_packet['spec'].shape[2], 72)) * 1e-6

    for station in range(0, np.size(obse_packet['spec'], axis=1)):
        # rotate the spectra back to true north
        obse_packet['ncSpec'][:, station, :, :], obse_packet['ncDirs'] = prepdata.grid2geo_spec_rotate(
                obse_packet['directions'],  obse_packet['spec'][:, station, :, :])
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
    # tempClass = prepdata.prepDataLib.PrepDataTools()
    gridPack = prepdata.makeCMSgridNodes(float(cio.sim_Packet[0]), float(cio.sim_Packet[1]),
                                                     float(cio.sim_Packet[2]), dep_pack['dx'], dep_pack['dy'],
                                                     dep_pack['bathy'])# dims [t, x, y]
    # ################################
    #        Make NETCDF files       #
    # ################################
    # STio = stwaveIO()
    if np.median(gridPack['elevation']) < 0:
        gridPack['elevation'] = -gridPack['elevation']

    fldrArch = os.path.join('waveModels', model, wave_version_prefix)
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
    fieldOfname = fileHandling.makeTDSfileStructure(os.path.join(Thredds_Base, 'Field'), fldrArch, datestring, field='Field')
    # make file name strings
    flagfname = os.path.join(fpath, 'Flags%s.out.txt' % datestring)  # startTime # the name of flag file
    fieldYaml = os.path.join('yaml_files',fldrArch, 'Field_globalmeta.yml')  # field
    varYaml = os.path.join('yaml_files',fldrArch, 'Field_var.yml')
    makenc.makenc_field(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)
    ###################################################################################################################
    ###############################   Plotting  Below   ###############################################################
    ###################################################################################################################
    dep_pack['bathy'] = np.transpose(dep_pack['bathy'], (0,2,1))  # dims [t, y, x]
    plotParams = [('waveHs', 'm'), ('bathymetry', 'NAVD88 $[m]$'), ('waveTp', 's'), ('waveDm', 'degTn')]
    if pFlag == True:
        for param in plotParams:
            print('    plotting field %s...' %param[0])
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
            sb.makeMovie(fpath + '/figures/CMTB_%s_%s_%s.mp4' % (wave_version_prefix, param[0], datestring), fList)
            [os.remove(ff) for ff in fList]

    ######################################################################################################################
    ######################################################################################################################
    ##################################  Wave Station Files HERE (loop) ###################################################
    ######################################################################################################################
    ######################################################################################################################
    print('stationList should be generated by look up table')
    # this is a list of file names to be made with station data from the parent simulation
    stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac_4.5m', 'adop-3.5m',
                   'xp200m', 'xp150m', 'xp125m']
    for gg, station in enumerate(stationList):

        stationName = 'CMTB-waveModels_CMS_%s_%s' % (wave_version_prefix, station)  # xp 125

        # this needs to be the same order as the run script
        stat_yaml_fname = os.path.join('yaml_files', fldrArch, 'Station_var.yml')
        globalyaml_fname = os.path.join('yaml_files', fldrArch, 'Station_globalmeta.yml')

        # getting lat lon, easting northing idx's
        Idx_i = len(gridPack['i']) - np.argwhere(gridPack['i'] == stat_packet['iStation'][
            gg]).squeeze() - 1  # to invert the coordinates from offshore 0 to onshore 0
        Idx_j = np.argwhere(gridPack['j'] == stat_packet['jStation'][gg]).squeeze()

        stat_data = {'time': nc.date2num(stat_packet['time'][:], units='seconds since 1970-01-01 00:00:00'),
                     'waveHs': stat_packet['waveHs'][-len(stat_packet['time'][:]):, gg],
                     'waveTm': np.ones_like(stat_packet['waveHs'][-len(stat_packet['time'][:]):, gg]) * -999,  # this isn't output by model, but put in fills to stay consistent
                     'waveDm': stat_packet['WaveDm'][-len(stat_packet['time'][:]):, gg],
                     'waveTp': stat_packet['Tp'][-len(stat_packet['time'][:]):, gg],
                     'waterLevel': stat_packet['waterLevel'][-len(stat_packet['time'][:]):, gg],
                     'swellHs': stat_packet['swellHs'][-len(stat_packet['time'][:]):, gg],
                     'swellTp': stat_packet['swellTp'][-len(stat_packet['time'][:]):, gg],
                     'swellDm': stat_packet['swellDm'][-len(stat_packet['time'][:]):, gg],
                     'seaHs': stat_packet['seaHs'][-len(stat_packet['time'][:]):, gg],
                     'seaTp': stat_packet['seaTp'][-len(stat_packet['time'][:]):, gg],
                     'seaDm': stat_packet['seaDm'][-len(stat_packet['time'][:]):, gg],
                     'station_name': stationName,
                     'directionalWaveEnergyDensity': obse_packet['ncSpec'][-len(stat_packet['time'][:]):, gg, :, :],
                     'waveDirectionBins': obse_packet['ncDirs'],
                     'waveFrequency': obse_packet['wavefreqbin'],
                     ###############################
                     'DX': dep_pack['dx'],
                     'DY': dep_pack['dy'],
                     'NI': dep_pack['NI'],
                     'NJ': dep_pack['NJ'],
                     'grid_azimuth': gridPack['azimuth']}

        w = go.getWaveSpec(station)
        try:
            stat_data['Latitude'] = w['lat']
            stat_data['Longitude'] = w['lon']
        except (TypeError, UnboundLocalError):
            stat_data['Latitude'] = -999   # gridPack['longitude'][Idx_i, Idx_j] # something seems wrong with gridPack
            stat_data['Longitude'] = -999  # gridPack['latitude'][Idx_i, Idx_j]

        ## make netCDF: Name files and make sure server directory has place for files to go
        print('making netCDF for model output at %s ' % station)
        outFileName = fileHandling.makeTDSfileStructure(os.path.join(Thredds_Base, station), fldrArch,  datestring, station)
        makenc.makenc_Station(stat_data, globalyaml_fname=globalyaml_fname, flagfname=flagfname,
                              ofname=outFileName, stat_yaml_fname=stat_yaml_fname)

        print("netCDF file's created for station: %s " % station)
        ###################################################################################################################
        ###############################   Plotting  Below   ###############################################################
        ###################################################################################################################

        if pFlag == True and w is not None and 'time' in w:
            if full == False:
                w['dWED'], w['wavedirbin'] = sbwave.HPchop_spec(w['dWED'], w['wavedirbin'], angadj=70)
            obsStats = sbwave.waveStat(w['dWED'], w['wavefreqbin'], w['wavedirbin'])

            modStats = sbwave.waveStat(obse_packet['ncSpec'][:, gg, :, :], obse_packet['wavefreqbin'],
                                       obse_packet['ncDirs'])  # compute model stats here

            time, obsi, modi = sb.timeMatch(nc.date2num(w['time'], 'seconds since 1970-01-01'),
                                            np.arange(w['time'].shape[0]),
                                            nc.date2num(stat_packet['time'][:], 'seconds since 1970-01-01'),
                                            np.arange(len(stat_packet['time'])))  # time match

            for param in modStats:  # loop through each bulk statistic
                if param in ['Tp', 'Tm10']:
                    units = 's'
                    title = '%s period' % param
                elif param in ['Hm0']:
                    units = 'm'
                    title = 'Wave Height %s ' % param
                elif param in ['Dm', 'Dp']:
                    units = 'degrees'
                    title = 'Direction %s' % param
                    # elif param in ['sprdF', 'sprdD']:
                    #     units = ''
                    #     title = 'Spread %s ' % param

                    # now run plots
                if param in ['Hm0', 'Tp', 'Tm', 'Tave', 'Tm10', 'Dp', 'Dm', 'Dm2']:
                    print('    plotting %s: %s' % (station, param))
                    p_dict = {'time': nc.num2date(time, 'seconds since 1970-01-01'),
                              'obs': obsStats[param][obsi.astype(int)],
                              'model': modStats[param][modi.astype(int)],
                              'var_name': param,
                              'units': units,  # ) -> this will be put inside a tex math environment!!!!
                              'p_title': title}
                    ofname = os.path.join(fpath, 'figures',
                                          'Station_%s_%s_%s.png' % (station, param, datestring))
                    # make sure
                    if len(p_dict['obs']) > 4 and not np.ma.array(p_dict['obs'], copy=False).mask.all():

                        stats = oP.obs_V_mod_TS(ofname, p_dict, logo_path='ArchiveFolder/CHL_logo.png')

                        if station == 'waverider-26m' and param == 'Hm0':
                            # this is a fail safe to abort run if the boundary conditions don't
                            # meet quality standards below
                            bias = 0.1  # bias has to be within 10 centimeters
                            RMSE = 0.1  # RMSE has to be within 10 centimeters
                            try:
                                assert stats['RMSE'] < RMSE, 'RMSE test on spectral boundary energy failed'
                                assert np.abs(stats['bias']) < bias, 'bias test on spectral boundary energy failed'
                            except:
                                print('!!!!!!!!!!FAILED BOUNDARY!!!!!!!!\ndeleting data from thredds!')
                                os.remove(fieldOfname)
                                os.remove(outFileName)
                                raise RuntimeError('The Model Is not validating its offshore boundary condition')

def CMSFanalyze(inputDict, cmsfio):
    """ This runs the analyze script for cmsflow

    This Function is the master call for the  data postprocessing for
    the Coastal Model Test Bed (CMTB).  It is designed to pull from
    GetData and utilize prep_datalib for development of the FRF CMTB

    Args:
        startTime: this is the start time of this model run
        inputDict: this is an input dictionary that was generated with the keys from the project input yaml file


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    # ___________________define Global Variables___________________________________
    pFlag = inputDict.get('plotFlag', True)
    # version prefixes!
    flow_version_prefix = cmsfio.version_prefix  # inputDict['version_prefix']
    path_prefix = inputDict['path_prefix']                   # for organizing data
    simulationDuration = inputDict['simulationDuration']
    Thredds_Base = inputDict.get('netCDFdir', '/home/%s/thredds_data/'.format(check_output('whoami', shell=True)[:-1]))
    d1 = DT.datetime.strptime(inputDict['startTime'], '%Y-%m-%dT%H:%M:%SZ')
    date_str = inputDict.get('datestring', d1.strftime('%Y%m%dT%H%M%SZ'))
    prepdata = prepDataLib.PrepDataTools()
    fpath = os.path.join(path_prefix, date_str)
    model = inputDict['modelSettings']['name'].lower() + 'f'
    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    t = DT.datetime.now()
    # load the files now, from the self.fname location assigned during input file write, will load to class attributes
    cmsfio.read_CMSF_all()
    print('Loading Files Took {} seconds '.format((DT.datetime.now() - t).seconds))

    # i think the only change i have to make is to convert the velocities into the same coord system as the gages?

    # IMPORTANT NEW INFORMATION!!
    # the output files DO NOT have all nodes, ONLY the nodes that are NOT -999
    # so I guess we save the not -999 nodes only?
    # it LOOKS LIKE the .xy file has the list of all the computed nodes in it, so we will use that to cross-reference?
    # is the order of the solution nodes the same as the .xy nodes? yes - according to CMS users manual DLY 06/18/2018.

    # OTHER NEW INFORMATION!!!
    # the nodes in the .xy file appear to be the non -999 cellID's in the .tel file in REVERSE!  so higher cellID's
    # are on top - see testing in DLY_telFile that I did on 6/19/2018.  THE "nodeIndex" in the xy_dict is NOT THE SAME
    # as the cellID's in the tel file!!!!!!!!!!!!!!
    if cmsfio.hotStartFlag is False:
        d1F = d1 - DT.timedelta(hours=int(24*cmsfio.durationRamp))  # convert days to hours
    else:
        d1F = d1
    # velocity info
    cmsfWrite = {'aveE': cmsfio.vel_dict['vx'].copy(), 'aveN': cmsfio.vel_dict['vy'].copy(),
                 'waterLevel': cmsfio.eta_dict['wl'], # -999 is masked
                 'coldStart': not cmsfio.hotStartFlag,   # want cold start not hot start, so take inverse
                 'time': nc.date2num(np.array([d1F + DT.timedelta(0, x * 3600, 0) for x in cmsfio.vel_dict['time']]), 'seconds since 1970-01-01')}

    # tstep = inputDict['flow_time_step']  NO NO NO NO!  OUTPUTS HOURLY REGARDLESS OF TIMESTEP!
    cmsfWrite = prepdata.modCMSFsolnDict(cmsfWrite, cmsfio.telnc_dict, simulationDuration)

    # now hand this to makenc_CMSFrun
    print('Writing simulation netCDF files.')
    ofname = fileHandling.makeTDSfileStructure(os.path.join(Thredds_Base, model), flow_version_prefix, date_str, 'field')

    # global yamls are version prefix specific
    ncGlobalYaml = os.path.join('yaml_files', 'flowModels', model, flow_version_prefix, 'CMSFrun_global.yml')
    ncVarYaml = os.path.join('yaml_files', 'flowModels', model, 'CMSFrun_var.yml')
    makenc.makenc_CMSFrun(os.path.join(Thredds_Base, model, flow_version_prefix, ofname), cmsfWrite, ncGlobalYaml, ncVarYaml)

    ###################################################################################################################
    ###############################   Plotting  Below   ###############################################################
    ###################################################################################################################
    # now make the plots off the cmsfWrite dictionary, create velocity magnitude dataset to plot
    cmsfWrite['vMag'] = np.sqrt(np.power(cmsfWrite['aveE'], 2) + np.power(cmsfWrite['aveN'], 2))

    # i think that cmsfWrite gets modified during makenc_CMSFrun?  so I need to put depth back into the
    # keys if it gets converted to elevation earlier
    if 'depth' not in cmsfWrite.keys():
        cmsfWrite['depth'] = -1*cmsfWrite['elevation']

    # mask all values where water level is -999
    maskInd = cmsfWrite['waterLevel'] == -999

    cmsfWrite['vMag'] = np.ma.masked_where(maskInd, cmsfWrite['vMag'])
    cmsfWrite['waterLevel'] = np.ma.masked_where(maskInd, cmsfWrite['waterLevel'])
    cmsfWrite['aveE'] = np.ma.masked_where(maskInd, cmsfWrite['aveE'])
    cmsfWrite['aveN'] = np.ma.masked_where(maskInd, cmsfWrite['aveN'])
    cmsfWrite['depth'] = np.ma.masked_where(cmsfWrite['depth'] == -999, cmsfWrite['depth'])

    # tack on the xFRF and yFRF values since I DONT have them in the solution netcdf files anymore
    cmsfWrite['xFRF'] = cmsfio.telnc_dict['xFRF']
    cmsfWrite['yFRF'] = cmsfio.telnc_dict['yFRF']

    # plotParams = [('WL', 'm', 'waterLevel'), ('depth', 'NAVD88 $[m]$', 'depth'), ('VelMag', 'm/s', 'vMag')]
    plotParams = [('WL', 'm', 'waterLevel'), ('VelMag', 'm/s', 'vMag')]
    if pFlag:
        ########################## spatial plots ########################################
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
                fnameSuffix = 'figures/CMTB_CMSF_%s_%s_%s.png' % (flow_version_prefix, param[0], numStr)
                oP.plotUnstructField(ofname=os.path.join(fpath, fnameSuffix), pDict=pDict)

            # now make a movie for each one, then delete pictures
            fList = sorted(glob.glob(fpath + '/figures/*%s*.png' % param[0]))
            sb.makeMovie(fpath + '/figures/CMTB_CMSF_%s_%s_%s.mp4' % (flow_version_prefix, param[0], date_str), fList)
            sb.myTarMaker(os.path.join(fpath, 'figures', param[0]+'individuals'), fList, removeFiles=True)

        ########################## station plots ########################################
        go = getDataFRF.getObs(d1, d2=d1+DT.timedelta(hours=simulationDuration))
        stationList = ['awac-11m', 'awac-6m', 'awac-4.5m', 'adop-3.5m']  # only list stations with current measurements
        exclude = set(string.punctuation)
        for station in stationList:
            # plot velocity plots [only]
            currents = go.getCurrents(station)
            if currents is not None:
                # find the closest node and pull that data
                ind, dist = gridTools.findNearestUnstructNode(currents['xFRF'], currents['yFRF'], cmsfWrite)
                matchedEpochTime, idxAvgObs, idxAvgMod = sb.timeMatch(currents['epochtime'], obs_data=None,
                                                                      model_time=cmsfWrite['time'], model_data=None)

                if len(matchedEpochTime) > 1:
                    # ave E velocity
                    path = fpath + '/figures/CMTB_CMSF_%s_%s_%s_aveE.png' % (flow_version_prefix, date_str, ''.join(ch for ch in station if ch not in exclude).strip())
                    p_dict = {'time': currents['time'][idxAvgObs],
                              'obs': currents['aveU'][idxAvgObs],
                              'model': cmsfWrite['aveE'][idxAvgMod, ind],
                              'var_name': '$\overline{U}$',
                              'units': 'm/s'}
                    oP.obs_V_mod_TS(path, p_dict)
                    # ave N velocity
                    path = fpath + '/figures/CMTB_CMSF_%s_%s_%s_aveN.png' % (flow_version_prefix, date_str, ''.join(ch for ch in station if ch not in exclude).strip())
                    p_dict = {'time': currents['time'][idxAvgObs],
                              'obs': currents['aveV'][idxAvgObs],
                              'model': cmsfWrite['aveN'][idxAvgMod, ind],
                              'var_name': '$\overline{V}$',
                              'units': 'm/s'}
                    oP.obs_V_mod_TS(path, p_dict)

            # what stations have water level?
            # if 'waverider' not in station:
            #
            #     oDict = getPlotData.CMSF_wlData(cmsfWrite, station)
            #     if oDict is None:
            #         print('Water level data missing for %s' % station)
            #         pass
            #
            #     else:
            #         if len(oDict['time']) <= 1:
            #             pass
            #         else:
            #             # water level
            #             fname = fpath + '/figures/CMTB_CMSF_%s_%s_%s_WL.png' % (flow_version_prefix, date_str, ''.join(ch for ch in station if ch not in exclude).strip())
            #             p_dict = {'time': oDict['time'],
            #                       'obs': oDict['obsWL'],
            #                       'model': oDict['modWL'],
            #                       'var_name': '$WL (NAVD88)$',
            #                       'units': 'm'}
            #             oP.obs_V_mod_TS(fname, p_dict)