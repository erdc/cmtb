# -*- coding: utf-8 -*-
"""
  This code is the front and backend of the STWAVE module for the coastal model test bed, the scheme was the first developed
    and code has since been updated to reflect process updates, however some inefficiencies still exist.  please
    feel free to make these better and share your work on the CMTB repo
    Thanks,
    Spicer
"""
# import libraries
import prepdata.prepDataLib
from testbedutils import anglesLib
from prepdata.inputOutput import stwaveIO
from plotting import operationalPlots as oP
from prepdata import inputOutput
from getdatatestbed import getDataFRF
import prepdata.prepDataLib as STPD
import datetime as DT
import getopt, glob, os, sys, shutil, makenc, warnings
import os
import netCDF4 as nc
import numpy as np
from testbedutils import geoprocess as gp
from testbedutils import sblib as sb


def STsimSetup(startTime, inputDict):
    """
    This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB).  It is designed to pull from GetData and utilize
    prep_datalib for development of the FRF CMTB
    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): input string in format YYYY-mm-ddThh:mm:ssZ
        inputDict (dict):  See documentation associated with input yamls
    Returns:
        nproc_parent (int): number of processors to run simultion for paret sim, will return -1 if this
            script things it should abort run
        nproc_nest (int): number of processors to run simulation in for nest, will return -1 is this script
            thinks it should abort run

    """
    # unpack dictionary -- Required
    version_prefix = inputDict['version_prefix']
    timerun =  inputDict['simulationDuration']
    plotFlag = inputDict['pFlag']
    background_grid_nested = inputDict['gridDEP_nested']
    background_grid_parent = inputDict['gridDEP_parent']
    if inputDict['workingDirectory'].endswith(inputDict['version_prefix']):
        path_prefix = inputDict['workingDirectory']
    else:
        path_prefix = os.path.join(inputDict['workingDirectory'],inputDict['version_prefix'] )
    if 'THREDDS' in inputDict:
        server = inputDict['THREDDS']
    else:
        print('Chosing CHL thredds by Default, this may be slower!')
        server = 'CHL'
    if 'ForcedSurveyDate' in inputDict:
        ForcedSurveyDate = inputDict['ForcedSurveyDate']
        FSDpieces = ForcedSurveyDate.split('-')
        ForcedSurveyDate = DT.datetime(int(FSDpieces[0]), int(FSDpieces[1]), int(FSDpieces[2]))
    else:
        ForcedSurveyDate = None
    if not os.path.isfile(background_grid_parent):
        raise EnvironmentError('check your background Grid parent file name')
    if not os.path.isfile(background_grid_nested):
        raise EnvironmentError('Check your Background grid Nested file name')
    # _________________________________________________________________________________________________________________
    # defaults of the setup
    TOD = 0 # hour of day simulation to start (UTC)
    FRFgaugelocsFile= 'ArchiveFolder/frf_sensor_locations.pkl'
    numNest = 3  # number of points to use as nesting seed
    model = 'STWAVE'
    # ______________________________________________________________________________
    # define version parameters
    if version_prefix == 'FP':
        print 'FP = Full plane Chosen'
        full = True # full plane
    elif version_prefix in ['HP', 'CBHP']:
        print 'HP = Half plane Chosen'
        full = False  # half plane
    elif version_prefix in ['CB', 'CBThresh', 'CBT1', 'CBT2']:
        print 'CB = cBathy Run chosen'
        background_grid_nested = inputDict['gridDEP_nested'].replace('5', '10') # make this dummy proof
        assert startTime[10] == 'T', 'End time for simulation runs must be in the format YYYY-MM-DDThh:mm:ssZ'
        full = False # cbathy simulations run in half plane
        waveHsThreshold = 1.2
        # if os.path.exists(pickleFname):
        #     os.remove(pickleFname)
    else:
        raise NameError('Need version prefix to run')

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
    d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    date_str = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime
    assert len(startTime) == 20 and startTime[10] == 'T' and startTime[-1] == 'Z', \
                'Your Time does not fit convention, check T/Z and input format, expects %Y-%m-%dT%H%M%SZ'
    if type(timerun) == str:
        timerun = int(timerun)
    # ______________________________________________________________________________
    # __________________Make Diretories_____________________________________________
    if not os.path.exists(os.path.join(path_prefix, date_str)):  # if it doesn't exist
        os.makedirs(os.path.join(path_prefix, date_str))  # make the directory
    if not os.path.exists(os.path.join(path_prefix ,date_str, "figures/")):
        os.makedirs(os.path.join(path_prefix, date_str, "figures/"))

    print "Model Time Start : %s  Model Time End:  %s" % (d1, d2)
    print u"files will be place in {0} folder".format(os.path.join(path_prefix, date_str))
    ###################################################################################################################
    #######################   Begin Gathering Data      ###############################################################
    ###################################################################################################################
    ## _____________WAVES____________________________
    print "_________________\nGathering Wave Data"
    # retrieve waves
    go = getDataFRF.getObs(d1, d2, THREDDS=server)
    try:
        rawspec = go.getWaveSpec(gaugenumber='waverider-26m')
        assert 'time' in rawspec, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (d1, d2)

    except AssertionError:
        rawspec = go.getWaveSpec(gaugenumber='waverider-17m')
        background_grid_parent = os.path.join(os.path.split(inputDict['gridDEP_parent'])[0], 'Regional_17mGrid_50m.dep')

    if 'time' not in rawspec:
        print "\n++++\nThere's STILL No Wave data between %s and %s \n++++\n" % (d1, d2)
        return -1, -1 ## abort runs
    # assert 'time' in rawspec, "\n++++\nThere's STILL No Wave data between %s and %s \n++++\n" % (d1, d2)
    prepdata = STPD.PrepDataTools()
    # rotate and lower resolution of directional wave spectra

    wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=date_str, plot=plotFlag, full=full, outputPath=path_prefix)
    print "number of wave records %d with %d interpolated points" % (np.shape(wavepacket['spec2d'])[0], sum(wavepacket['flag']))
    # ____________ BATHY ______________________

    print '\n____________________\nGetting Bathymetric Data\n'
    stio = inputOutput.stwaveIO('')  # initializing io here so grid text can be written out
    # load grids to interp to STwAVE
    gtb = getDataFRF.getDataTestBed(d1, d2)  # this should be relocated to operational servers
    ofnameDep = os.path.join(path_prefix, date_str, '{}nested.dep'.format(date_str))
    # warnings.warn('GetData bathy is in get model data!')
    bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate)
    gridNodesNested = prepdata.GetOriginalGridFromSTWAVE(background_grid_nested[:-4]+'.sim', background_grid_nested)

    if version_prefix in ['FP', 'HP', 'CBHP']:
        # get data first
        bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate)
        # first find the nodes of the grid
        gridName='version_%s_SurveyDate_%s_SurveyNumber_%d' %(version_prefix, bathy['time'].strftime('%Y-%m-%d'), bathy['surveyNumber'])

    elif version_prefix == 'CB':
        bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate, cBKF=True)
        gridName='version_{}_SurveyDate_{}'.format(version_prefix, bathy['time'].strftime('%Y-%m-%dT%H%M%SZ'))

    elif version_prefix == 'CBThresh':
        bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate, cBKF_T=True)
        gridName='version_{}_SurveyDate_{}'.format(version_prefix, bathy['time'].strftime('%Y-%m-%dT%H%M%SZ'))

    print 'Sim start: %s\nSim End: %s\nSim bathy chosen: %s' % (d1, d2, bathy['time'])
    NestedBathy = prepdata.prep_Bathy(bathy, gridNodesNested, gridName=gridName, positiveDown=True)  # prep the grid to match the STWAVE domain in example grid file

    ## _____________WINDS______________________
    print '_________________\nGetting Wind Data'
    try:
        rawwind = go.getWind(gaugenumber=0)
        # average and rotate winds
        windpacket = prepdata.prep_wind(rawwind, wavepacket['epochtime'], maxdeadrecord=6)
        # wind height correction
        print 'number of wind records %d with %d interpolated points' % (
            np.size(windpacket['time']), sum(windpacket['flag']))
    except (RuntimeError, TypeError):
        windpacket = None
        print ' NO WIND ON RECORD'

    ## ___________WATER LEVEL__________________
    print '_________________\nGetting Water Level Data'
    try:
        # get water level data
        rawWL = go.getWL()
        # average WL
        WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])
        print 'number of WL records %d, with %d interpolated points' % (
                np.size(WLpacket['time']), sum(WLpacket['flag']))
    except (RuntimeError, TypeError):
        WLpacket = None

    ## ___________CURRENTS_____________________
    # print '______________\nGetting Currents'
    curpacket = None

    # ##################################################################
    # check data do you have any problems
    print 'Running Data Check\n+++++++++++++++++++++++++'
    prepdata.data_check(wavepacket, windpacket, WLpacket, curpacket)
    ##___________________________________________________________________________
    ##  Get sensor locations and add to sim file start
    loc_dict = go.get_sensor_locations(datafile=FRFgaugelocsFile, window_days=14)
    statloc =  []
    for gauge in loc_dict.keys():
        coords = loc_dict[gauge]
        try:
            statloc.append([coords['spE'], coords['spN']])
        except KeyError:
            continue
    if (version_prefix == 'CB' or version_prefix == 'CBHP') and \
            (d1 >= DT.datetime(2015,10,15) and d2 < DT.datetime(2015, 11, 1)):
        locD11 = go.getBathyDuckLoc(11)
        loc11 = np.array([[locD11['StateplaneE'], locD11['StateplaneN']]])
        locD12 = go.getBathyDuckLoc(12)
        loc12 = np.array([[locD12['StateplaneE'], locD12['StateplaneN']]])
        locD13 = go.getBathyDuckLoc(13)
        loc13 = np.array([[locD13['StateplaneE'], locD13['StateplaneN']]])
        locD14 = go.getBathyDuckLoc(14)
        loc14 = np.array([[locD14['StateplaneE'], locD14['StateplaneN']]])
        locD21 = go.getBathyDuckLoc(21)
        loc21 = np.array([[locD21['StateplaneE'], locD21['StateplaneN']]])
        locD22 = go.getBathyDuckLoc(22)
        loc22 = np.array([[locD22['StateplaneE'], locD22['StateplaneN']]])
        locD23 = go.getBathyDuckLoc(23)
        loc23 = np.array([[locD23['StateplaneE'], locD23['StateplaneN']]])
        locD24 = go.getBathyDuckLoc(24)
        loc24 = np.array([[locD24['StateplaneE'], locD24['StateplaneN']]])
        newloc = np.concatenate([loc11, loc12, loc13, loc14, loc21, loc22, loc23, loc24], axis=0)
        statloc = np.append(statloc, newloc, axis=0)
    statloc = np.array(statloc)
    # go get the nesting points
    whichpointsy = np.linspace(gridNodesNested['yFRF'][0], gridNodesNested['yFRF'][-1], numNest)
    nestLocDict = {} #initalize nesting output locations
    for key in whichpointsy:
        nestLocDict[key]= {'xFRF': gridNodesNested['xFRF'].max(),'yFRF':key}

    ###################################################################################################################
    #######################   Begin Writing Files   ###################################################################
    ###################################################################################################################
    # Last thing to do ... write files file
    print 'WRITING simulation Files'
    stio.write_dep(ofnameDep, NestedBathy)
    # now copy the outer domain to the local directory
    inputOutput.write_flags(date_str, path_prefix, wavepacket, windpacket, WLpacket, curpacket, gridFlag=False)
    shutil.copy2(background_grid_parent, os.path.join(path_prefix, date_str, '%s.dep' % date_str))
    stio.write_spec(date_str, path=path_prefix, STwavepacket=wavepacket)
    nproc_parent = stio.write_sim(date_str, path=path_prefix, snapbase=wavepacket['snapbase'], nested=0,
                                  windpacket=windpacket, WLpacket=WLpacket, curpacket=curpacket, statloc=statloc,
                                  full=full, version_prefix=version_prefix, nestpoints=nestLocDict)

    nproc_nest = stio.write_sim(date_str, path=path_prefix, snapbase=wavepacket['snapbase'],
                                nested=1, windpacket=windpacket, WLpacket=WLpacket, curpacket=curpacket,
                                statloc=statloc, full=full)

    return nproc_parent, nproc_nest

def STanalyze(startTime, inputDict):
    """the master call for the simulation post process for STWAVE
    plots and netCDF files are made at request

    This is the bulk of the function that analyzes the STWAVE
            output from the CMTB
    Args:
        inputDict (dict): dictionary of input parameters read in by input file
        startTime (str):  a string that has date in it by which the
            end of the run is designated ex: '2015-12-25T00:00:00Z'
    Returns:
          None

    """
    # ___________________ Unpack input dictionary _________________________________
    # _____________________________________________________________________________
    # globals
    model = inputDict['model']
    angadj = 70  # angle to adjust the
    # unpack dictionary
    version_prefix = inputDict['version_prefix']
    timerun = inputDict['simulationDuration']
    plotFlag = inputDict['pFlag']
    Thredds_Base = inputDict['netCDFdir']
    if inputDict['workingDirectory'].endswith(inputDict['version_prefix']):
        path_prefix = inputDict['workingDirectory']
    else:
        path_prefix = os.path.join(inputDict['workingDirectory'],inputDict['version_prefix'] )
    if 'THREDDS' in inputDict:
        server = inputDict['THREDDS']
    else:
        print('Chosing CHL thredds by Default, this may be slower!')
        server = 'CHL'
    ######### model static global setups
    if type(timerun) == str:
        timerun = int(timerun)
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    try:
        startTime = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    except:
        raise EnvironmentError('Your Time does not fit convention, please use "YYYY-MM-DDThh:mm:ssZ" in UTC')

    endTime = startTime + DT.timedelta(0, timerun * 3600, 0)
    datestring = startTime.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, datestring)
    fldrArch = os.path.join(model, version_prefix)
    print('\nBeggining of Analyze Script\nLooking for file in {}'.format(fpath))
    print('\nData Start: {}  Finish: {}'.format(startTime, endTime))

    go = getDataFRF.getObs(startTime, endTime, THREDDS=server)  # setting up get data instance
    prepdata = STPD.PrepDataTools()  # initializing instance for rotation scheme
    #################################################################################################
    #################################################################################################
    ##################### Begin by Loading Data from Simulation #####################################
    #################################################################################################
    #################################################################################################

    stio = stwaveIO(fpath)  # =pathbase) looks for model output files in folder to analyze
    assert len(stio.obsefname) != 0, 'There are no data to load for %s' %startTime
    print('Loading Statistic Files ....')
    d = DT.datetime.now()  # for data load timing

    stat_packet = stio.statload(nested=0)  # loads modeled data (parent station file)
    # correct model outout angles from STWAVE(+CCW) to Geospatial (+CW)
    stat_packet['WaveDm'] = anglesLib.STWangle2geo(stat_packet['WaveDm'])
    stat_packet['Udir'] = anglesLib.STWangle2geo(stat_packet['Udir']) # wind direction
    # correct angles
    stat_packet['WaveDm'] = anglesLib.angle_correct(stat_packet['WaveDm'])
    stat_packet['Udir'] = anglesLib.angle_correct(stat_packet['Udir']) # wind direction

    try:
        # try to load the nested station file
        modelpacket_nest = stio.statload(nested=1)
        nest = 1
        # correct model output angles from STWAVE(+CCW) to Geospatial (+CW)
        modelpacket_nest['WaveDm'] = anglesLib.STWangle2geo(modelpacket_nest['WaveDm'])  # convert STW angles to MET out
        modelpacket_nest['Udir'] = anglesLib.STWangle2geo(modelpacket_nest['Udir'])  # wind dierction
        # correct angles
        modelpacket_nest['WaveDm'] = anglesLib.angle_correct(modelpacket_nest['WaveDm'])
        modelpacket_nest['Udir'] = anglesLib.angle_correct(modelpacket_nest['Udir']) # wind direction
    except IndexError:
        nest = 0
    # Load Spectral Data sets
    obse_packet = stio.obseload(nested=False)
    obse_nested = stio.obseload(nested=True)

    # creating template for spec to go into netCDF files
    obse_packet['ncSpec'] = np.ones((obse_packet['spec'].shape[0], obse_packet['spec'].shape[1], obse_packet['spec'].shape[2], 72)) * 1e-6
    obse_nested['ncSpec'] = np.ones((obse_nested['spec'].shape[0], obse_nested['spec'].shape[1], obse_nested['spec'].shape[2], 72)) * 1e-6
    # rotating the spectra back to true north (TN) then
    for station in range(0, np.size(obse_packet['spec'], axis=1)):
        obse_packet['ncSpec'][:, station, :, :], obse_packet['ncDirs'] = prepdata.grid2geo_spec_rotate(
            obse_packet['directions'], obse_packet['spec'][:, station, :, :])
        obse_nested['ncSpec'][:, station, :, :], obse_nested['ncDirs'] = prepdata.grid2geo_spec_rotate(
            obse_nested['directions'], obse_nested['spec'][:, station, :, :])
        # converting m^2/Hz/radians back to m^2/Hz/degree
        # note that units of degrees are on the denominator which requires a deg2rad conversion instead of rad2deg
        obse_packet['ncSpec'][:, station, :, :] = np.deg2rad(obse_packet['ncSpec'][:, station, :, :])
        obse_nested['ncSpec'][:, station, :, :] = np.deg2rad(obse_nested['ncSpec'][:, station, :, :])
    if obse_packet['spec'].shape[3] == 72:
        full = True
    else:
        full = False

 ######################################################################################################################
 ######################################################################################################################
 ##################################  Spatial Data HERE     ############################################################
 ######################################################################################################################
 ######################################################################################################################
    # load Files
    print '  ..begin loading spatial files ....'
    rad_nest = stio.genLoad('rad', nested=True)
    break_nest = stio.genLoad('break', nested=True)
    dep_pack = prepdata.GetOriginalGridFromSTWAVE(stio.simfname[0], stio.depfname[0])
    dep_nest = prepdata.GetOriginalGridFromSTWAVE(stio.simfname_nest[0], stio.depfname_nest[0])
    Tp_pack = stio.TPload(nested=0) # this function is currently faster than genLoad
    # Tp_pack2 = stio.genLoad('waveTp', nested=False)
    Tp_nest = stio.TPload(nested=1)  # this function is currently faster than genLoad
    #  Tp_nest2 = stio.genLoad('waveTp', nested=True)
    wave_pack = stio.waveload(nested=0)
    # wave_pack2 = stio.genLoad('wave', nested=False)
    wave_nest = stio.waveload(nested=1)
    # wave_nest2 = stio.genLoad('wave', nested=True)


    print 'Files loaded, in %s ' %(DT.datetime.now() - d)

    if plotFlag == True:
        print "   BEGIN PLOTTING "
        d = DT.datetime.now()
        plotFnameRegional = 'figures/CMTB_waveModels_STWAVE_%s_Regional-' % version_prefix
        plotFnameLocal = 'figures/CMTB_waveModels_STWAVE_%s_Local-' % version_prefix
        ## first make dicts for nested plots
        dep_nest_plot = {'title': 'Local FRF Property: Bathymetry', 'xlabel': 'Longshore distance [m]',
                         'ylabel': 'Cross-shore distance [m]',      'field': dep_nest['bathy'],
                         'xcoord': dep_nest['xFRF'],                'ycoord': dep_nest['yFRF'],
                         'cblabel': 'Water Depth - NAVD88 $[m]$',   'time': dep_nest['time']}
        Hs_nest_plot = {'title': 'Local FRF North Property: Significant wave height $H_s$',
                        'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                        'field': wave_nest['Hs_field'],     'xcoord': dep_nest['xFRF'],
                        'ycoord': dep_nest['yFRF'],         'cblabel': 'Wave height $H_s [m]$',
                        'time': wave_nest['time']}
        Tm_nest_plot = {'title': 'Local FRF North Property: Mean wave period $T_m$',
                        'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                        'field': wave_nest['Tm_field'],     'xcoord': dep_nest['xFRF'],
                        'ycoord': dep_nest['yFRF'],         'cblabel': 'Mean Period $T_m [s]$',
                        'time': wave_nest['time']}
        # Dm_nest_plot = {'title': 'Local FRF North Property: Mean wave direction $D_m$',
        #                 'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
        #                 'field': wave_nest['Dm_field'],     'xcoord': dep_nest['xFRF'],
        #                 'ycoord': dep_nest['yFRF'],         'cblabel': 'Mean Direction $\degree Shore Normal$',
        #                 'time': wave_nest['time']}
        Tp_nest_plot = {'title': 'Local FRF North Property: Peak wave period $T_p$',
                        'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                        'field': Tp_nest['Tp_field'],       'xcoord': dep_nest['xFRF'],
                        'ycoord': dep_nest['yFRF'],         'cblabel': 'Peak Period $T_p [s]$',
                        'time': Tp_nest['time']}
        break_nest_plot = {'title': 'Local FRF North Property: Wave Dissipation',
                        'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                        'field': break_nest['dissipation'],       'xcoord': dep_nest['xFRF'],
                        'ycoord': dep_nest['yFRF'],         'cblabel': 'Wave Dissipation',
                        'time': Tp_nest['time']}
        # now create pickle if one's not around
        rads_nest_plot_x = {'title': 'Local FRF North Property: Radiation Stress Gradients - X',
                        'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                        'field': rad_nest['xRadGrad'],       'xcoord': dep_nest['xFRF'],
                        'ycoord': dep_nest['yFRF'],         'cblabel': 'Radiation Stress Gradients - X ',
                        'time': Tp_nest['time']}
        rads_nest_plot_y = {'title': 'Local FRF North Property: Radiation Stress Gradients - Y',
                            'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                            'field': rad_nest['yRadGrad'], 'xcoord': dep_nest['xFRF'],
                            'ycoord': dep_nest['yFRF'], 'cblabel': 'Radiation Stress Gradients - Y',
                            'time': Tp_nest['time']}
        # make nested plots
        oP.plotSpatialFieldData(dep_nest, dep_nest_plot, os.path.join(fpath, plotFnameLocal + 'bathy'), nested=True)
        oP.plotSpatialFieldData(dep_nest, Tp_nest_plot, os.path.join(fpath, plotFnameLocal + 'Tp'), nested=True)
        oP.plotSpatialFieldData(dep_nest, Hs_nest_plot, os.path.join(fpath, plotFnameLocal + 'Hs'), nested=True, directions=wave_nest['Dm_field'])
        oP.plotSpatialFieldData(dep_nest, Tm_nest_plot, os.path.join(fpath, plotFnameLocal + 'Tm'), nested=True)
        # oP.plotSpatialFieldData(dep_nest, Dm_nest_plot, plotFnameLocal + 'Dm', fpath, nested=True)
        oP.plotSpatialFieldData(dep_nest, rads_nest_plot_y, os.path.join(fpath, plotFnameLocal + 'xRG'), nested=True)
        oP.plotSpatialFieldData(dep_nest, rads_nest_plot_x, os.path.join(fpath, plotFnameLocal + 'yRG'), nested=True)
        oP.plotSpatialFieldData(dep_nest, break_nest_plot, os.path.join(fpath, plotFnameLocal + 'break'), nested=True)

        # find xfshore profile loc
        xshoreCoord = np.argmin(np.abs(dep_nest['yFRF']  - 945))
        for ttime in range(wave_nest['Hs_field'].shape[0]):
            fname = os.path.join(fpath, plotFnameLocal+ 'LocalxShoreWaveHeight_{}.png'.format(wave_nest['time'][ttime].strftime("%Y%m%dT%H%M%SZ")))
            oP.plotWaveProfile(dep_nest['xFRF'], wave_nest['Hs_field'][ttime, xshoreCoord,:], -dep_nest['bathy'][0,xshoreCoord,:], fname)
        ###########################
        #
        # # now make dicts for parent plots
        #
        ###########################
        dep_parent_plot = {'title': 'Regional Grid: Bathymetery',  'xlabel': 'Longshore distance [m]',
                           'ylabel': 'Cross-shore distance [m]',   'field': dep_pack['bathy'],
                           'xcoord': dep_pack['xFRF'],             'ycoord': dep_pack['yFRF'],
                           'cblabel': 'Water Depth - NAVD88 $[m]$','time': dep_pack['time']}
        Tp_parent_plot = {'title': 'Regional Grid: Peak wave period $T_p$',
                          'xlabel': 'Longshore distance [m]',  'ylabel': 'Cross-shore distance [m]',
                          'field': Tp_pack['Tp_field'], 'xcoord': dep_pack['xFRF'],
                          'ycoord': dep_pack['yFRF'],   'cblabel': 'Peak Period $T_p [s]$',
                          'time': Tp_pack['time']}
        Hs_parent_plot = {'title': 'Regional Grid: Significant wave height $H_s$',
                          'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                          'field': wave_pack['Hs_field'],     'xcoord': dep_pack['xFRF'],
                          'ycoord': dep_pack['yFRF'],         'cblabel': 'Wave Height $H_s [m]$',
                          'time': wave_pack['time']}
        Tm_parent_plot = {'title': 'Regional Grid: Mean wave period $T_m$',
                          'xlabel': 'Longshore distance [m]', 'ylabel': 'Cross-shore distance [m]',
                          'field': wave_pack['Tm_field'],     'xcoord': dep_pack['xFRF'],
                          'ycoord': dep_pack['yFRF'],         'cblabel': 'Mean Period $T_m [s]$',
                          'time': wave_pack['time']}
        # Dm_parent_plot = {'title': 'Regional Grid: Mean wave direction $D_m$',
        #                   'xlabel': 'Longshore distance [m]',  'ylabel': 'Cross-shore distance [m]',
        #                   'field': wave_pack['Dm_field'],      'xcoord': dep_pack['xFRF'],
        #                   'ycoord': dep_pack['yFRF'],          'cblabel': 'Mean Direction $\degree Shore Normal$',
        #                   'time': wave_pack['time']}

        oP.plotSpatialFieldData(dep_pack, dep_parent_plot, os.path.join(fpath, plotFnameRegional + 'bathy'), nested=0)
        oP.plotSpatialFieldData(dep_pack, Tm_parent_plot, prefix=os.path.join(fpath, plotFnameRegional + 'Tm'), nested=0)
        # oP.plotSpatialFieldData(dep_pack, Dm_parent_plot, plotFnameRegional + 'Dm', fpath, nested=0)
        oP.plotSpatialFieldData(dep_pack, Hs_parent_plot, os.path.join(fpath, plotFnameRegional + 'Hs'), nested=0, directions=wave_pack['Dm_field'])
        oP.plotSpatialFieldData(dep_pack, Tp_parent_plot, os.path.join(fpath, plotFnameRegional + 'Tp'), nested=0)

        # ################################
        # Make GIFs from Images          #
        # ## #############################

        # localDm = sorted(glob.glob(os.path.join(fpath,'figures/*Local-Dm*.png')))
        # sb.makegif(localDm, fpath+ plotFnameLocal + 'Dm_%s.gif' %(datestring))
        # [os.remove(ff) for ff in localDm]
        localTm = sorted(glob.glob(fpath + '/figures/*Local-Tm*.png'))
        localHs = sorted(glob.glob(fpath + '/figures/*Local-Hs*.png'))
        localTp = sorted(glob.glob(fpath + '/figures/*Local-Tp*.png'))
        localxHs = sorted(glob.glob(fpath + '/figures/*LocalxShoreWaveHeight*.png'))
        localxRad = sorted(glob.glob(fpath + '/figures/*Local-xRG*.png'))
        localyRad = sorted(glob.glob(fpath + '/figures/*Local-yRG*.png'))
        localBreak = sorted(glob.glob(fpath + '/figures/*Local-break*.png'))
        regTm = sorted(glob.glob(fpath + '/figures/*Regional-Tm*.png'))
        regHs = sorted(glob.glob(fpath + '/figures/*Regional-Hs*.png'))
        regTp = sorted(glob.glob(fpath + '/figures/*Regional-Tp*.png'))

        sb.makegif(localTm, os.path.join(fpath, plotFnameLocal + 'Tm_%s.gif'%(datestring)))
        sb.makegif(localHs, os.path.join(fpath, plotFnameLocal + 'Hs_%s.gif' %(datestring)))
        sb.makegif(localTp, os.path.join(fpath, plotFnameLocal + 'Tp_%s.gif' %(datestring)))
        sb.makegif(localxHs, os.path.join(fpath, plotFnameLocal + 'xShoreWaveHeight_%s.gif' %(datestring)))
        sb.makegif(localxRad, os.path.join(fpath, plotFnameLocal + 'xRadGrad_{}.gif'.format(datestring)))
        sb.makegif(localyRad, os.path.join( fpath, plotFnameLocal + 'yRadGrad_{}.gif'.format(datestring)))
        sb.makegif(localBreak, os.path.join(fpath, plotFnameLocal + 'Break_{}.gif'.format(datestring)))
        sb.makegif(regTm, os.path.join(fpath, plotFnameRegional +  'Tm_%s.gif' %(datestring)))
        sb.makegif(regHs, os.path.join(fpath, plotFnameRegional + 'Hs_%s.gif' %(datestring)))
        sb.makegif(regTp, os.path.join(fpath, plotFnameRegional + 'Tp_%s.gif' %(datestring)))

        [os.remove(ff) for ff in localHs]
        [os.remove(ff) for ff in localTp]
        [os.remove(ff) for ff in localxHs]
        [os.remove(ff) for ff in localxRad]
        [os.remove(ff) for ff in localyRad]
        [os.remove(ff) for ff in localBreak]
        [os.remove(ff) for ff in regTm]
        [os.remove(ff) for ff in regHs]
        [os.remove(ff) for ff in regTp]
        [os.remove(ff) for ff in localTm]

        # regDm = sorted(glob.glob(fpath + '/figures/*Regional-Dm*.png'))
        # sb.makegif(regDm, fpath+ plotFnameRegional + 'Dm_%s.gif' %(datestring))
        # [os.remove(ff) for ff in regDm]
        print '-- Spatial plots were made in %s ' %(DT.datetime.now() - d)

    # ################################
    # Make NETCDF files              #
    # ## #############################
    #  local grid global metadata
    locGlobYml = 'yaml_files/waveModels/{}/{}/Field_Local_{}_globalmeta.yml'.format(model, version_prefix, version_prefix)
    # Regional grid Global metadata
    regGlobYml = 'yaml_files/waveModels/{}/{}/Field_Regional_{}_globalmeta.yml'.format(model, version_prefix, version_prefix)
    globalyaml_fname_station = 'yaml_files/waveModels/{}/{}/Station_{}_globalmeta.yml'.format(model, version_prefix, version_prefix)
    station_var_yaml = 'yaml_files/waveModels/{}/Station_Directional_Wave_var.yml'.format(model)
    locVarYml =  'yaml_files/waveModels/{}/Field_var.yml'.format(model)
    regVarYml = 'yaml_files/waveModels/{}/Field_var.yml'.format(model)
    flagfname = fpath + '/Flags%s.out.txt' % datestring  # startTime # the name of flag file
    if np.median(dep_pack['bathy']) < 0:
        dep_pack['bathy'] = - dep_pack['bathy']
    if np.median(dep_nest['bathy']) < 0:
        dep_nest['bathy'] = -dep_nest['bathy']

    if dep_pack['gridFname'].lower().strip('"')=='regional_17mgrid_50m':
        # check domain on server - assume that it's correct
        def Fixup17mGrid(version_prefix, wave_pack, dep_pack, Tp_pack, fill_value=np.nan):
            """This function is designed to add filler data to size the truncated grid that's created with the offshore
            boundary is initalized from the 17m waverider not the 26m waverider,  data output from the model are placed
            into filled arrays with the fill value default in this function

            Args:
                version_prefix (str): version prefix associated with output data, function uses this to pull full
                    grid data (shape, cell coordinates, etc)
                wave_pack (dict): will look for
                dep_pack (dict):
                Tp_pack (dict):
                fill_value: will use this fill value to fill arrays where the model did not produce data (defaule=np.nan)

            Returns:
                new dictionaries that are the size of the full data arrays, using fill values

            """
            ncfile = nc.Dataset('http://bones/thredds/dodsC/cmtb/waveModels/STWAVE/{}/Regional-Field/Regional-Field.ncml'.format(version_prefix))
            NJ, NI =  ncfile['waveHs'].shape[1:]
            fill = np.ones((wave_pack['Hs_field'].shape[0], NJ, NI)) * fill_value
            # replace all the field values with the filled, value to the same dim
            # start with wave_pack
            for var in wave_pack.keys():
                if var.split('_')[-1].lower() == 'field':
                    fill[:, :, slice(0, wave_pack[var].shape[2])] = wave_pack[var]
                    wave_pack[var] = fill
            # fill dep_pack
            for var in dep_pack.keys():
                if var.split('_')[-1].lower() == 'bathy':
                    fill[:, :, slice(0, dep_pack[var].shape[2])] = dep_pack[var]
                    dep_pack[var] = fill
                elif var in ['yFRF', 'xFRF']:
                    dep_pack[var] = ncfile[var][:]
            try:
                dep_pack['longitude'] = ncfile['longitude'][:]
                dep_pack['latitude'] = ncfile['latitude'][:]
            except IndexError:
                pass
            dep_pack['NI'] = NI  # reset the dimensions so its written properly
            dep_pack['NJ'] = NJ  # reset dims
            # finally Tp
            fill[:, :, slice(0, Tp_pack['Tp_field'].shape[2])] = Tp_pack['Tp_field']
            Tp_pack['Tp_field'] = fill

            return wave_pack, dep_pack, Tp_pack
        wave_pack, dep_pack, Tp_pack = Fixup17mGrid(version_prefix, wave_pack, dep_pack, Tp_pack)
    # writing data librarys
    regionalDataLib = {'time': nc.date2num(wave_pack['time'], units='seconds since 1970-01-01 00:00:00'),
                       'station_name': 'Regional Simulation Field Data',
                       'waveHs': wave_pack['Hs_field'],
                       'waveTm': wave_pack['Tm_field'],
                       'waveDm': wave_pack['Dm_field'],
                       'waveTp': Tp_pack['Tp_field'],
                       'bathymetry': dep_pack['bathy'],
                       'latitude': dep_pack['latitude'],
                       'longitude': dep_pack['longitude'],
                       'xFRF': dep_pack['xFRF'],
                       'yFRF': dep_pack['yFRF'],
                       'DX': dep_pack['DX'],
                       'DY': dep_pack['DY'],
                       'NI': dep_pack['NI'],
                       'NJ': dep_pack['NJ']}

    # make regional field nc file, set file to be made on thredds
    # TdsFldrBase = Thredds_Base + '/'+ fpath.split('/')[2] + '/Regional-Field'
    TdsFldrBase = os.path.join(Thredds_Base,fldrArch)
    NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Regional-Field'), datestring, model=model)
    NCname = 'CMTB-waveModels_{}_{}_Regional-Field_{}.nc'.format(model, version_prefix, datestring)
    regionalOFName = os.path.join(NCpath, NCname)  # TdsF
    if not os.path.exists(os.path.join(TdsFldrBase, 'Regional-Field')):
        os.makedirs(os.path.join(TdsFldrBase, 'Regional-Field'))  # maameke the directory for the thredds data output
    if not os.path.exists(os.path.join(TdsFldrBase, 'Regional-Field', 'Regional-Field.ncml')):
        inputOutput.makencml(os.path.join(TdsFldrBase, 'Regional-Field', 'Regional-Field.ncml'))  # remake the ncml if its not there

    assert os.path.isfile(regGlobYml), 'NetCDF yaml files are not created'
    makenc.makenc_field(data_lib=regionalDataLib, globalyaml_fname=regGlobYml, flagfname=flagfname,
                        ofname=regionalOFName, var_yaml_fname=regVarYml)

    # Making record of the Date of the survey/inversion in datetime format
    gridNameSplit = dep_nest['gridFname'].split('_')
    if 'SurveyDate' in gridNameSplit:
        bathyDateA = gridNameSplit[np.argwhere(np.array(gridNameSplit[:]) == 'SurveyDate').squeeze() + 1].strip('"')
        # making the bathymetry grid
        try:
            bathyTime = DT.datetime.strptime(bathyDateA, '%Y-%m-%dT%H%M%SZ')
        except ValueError:
            bathyTime = DT.datetime.strptime(bathyDateA, '%Y-%m-%d')

    localDataLib = {'time': nc.date2num(wave_nest['time'], units='seconds since 1970-01-01 00:00:00'),
                    'station_name': 'Nested Simulation Field Data',
                    'waveHs': wave_nest['Hs_field'],
                    'waveTm': wave_nest['Tm_field'],
                    'waveDm': wave_nest['Dm_field'],
                    'waveTp': Tp_nest['Tp_field'],
                    'bathymetry': dep_nest['bathy'],
                    'xRadGrad': rad_nest['xRadGrad'],
                    'yRadGrad': rad_nest['yRadGrad'],
                    'dissipation': break_nest['dissipation'],
                    'bathymetryDate': nc.date2num(bathyTime, units='seconds since 1970-01-01 00:00:00'),
                    'latitude': dep_nest['latitude'],
                    'longitude': dep_nest['longitude'],
                    'xFRF': dep_nest['xFRF'],
                    'yFRF': dep_nest['yFRF'],
                    'NI': dep_nest['NI'],
                    'NJ': dep_nest['NJ'],
                    'DX': dep_nest['DX'],
                    'DY': dep_nest['DY']}

    # make local field nc file
    TdsFldrBase = os.path.join(Thredds_Base,fldrArch)
    NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Local-Field'), datestring, model=model)
    NCname = 'CMTB-waveModels_{}_{}_Local-Field_{}.nc'.format(model, version_prefix, datestring)
    localOFName = os.path.join(NCpath, NCname)  # Td

    if not os.path.exists(os.path.join(TdsFldrBase, 'Local-Field')):
        os.makedirs(os.path.join(TdsFldrBase, 'Local-Field')) # maameke the directory for th
    if not os.path.exists(os.path.join(TdsFldrBase, 'Local-Field', 'Local-Field.ncml')):
        inputOutput.makencml(os.path.join(TdsFldrBase, 'Local-Field', 'Local-Field.ncml'))
    makenc.makenc_field(data_lib=localDataLib, globalyaml_fname=locGlobYml, flagfname=flagfname,
                        ofname=localOFName, var_yaml_fname=locVarYml)
    ######################################################################################################################
    ######################################################################################################################
    ##################################  Wave Station Files HERE (loop) ###################################################
    ######################################################################################################################
    ######################################################################################################################
    RegionalStations = ['waverider-26m', 'waverider-17m', 'awac-11m', ':', ':', ':', ':', ':', ':', ':'] # where :'s are for sims in the nested
    NestedStations = [':', ':', ':','8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m', ]
    if (version_prefix == 'CB' or version_prefix == 'CBHP') and \
            (startTime >= DT.datetime(2015,10,15) and endTime < DT.datetime(2015, 11, 1)):
        NestedStations.extend(('Station_p11', 'Station_p12', 'Station_p13', 'Station_p14', 'Station_p21', 'Station_p22',
                              'Station_p23', 'Station_p24'))
    # writing station files from regional/parent simulation
    for gg, station in enumerate(RegionalStations):
        stat_yaml_fname = station_var_yaml
        if station != ':':
            # getting lat lon
            coords = gp.FRFcoord(stat_packet['Easting'][0, gg], stat_packet['Northing'][0, gg])
            stat_data = {'time': nc.date2num(stat_packet['time'][:], units='seconds since 1970-01-01 00:00:00'),
                         'waveHs': stat_packet['Hs'][:,gg],
                         'waveTm': stat_packet['Tm'][:, gg],
                         'waveDm': stat_packet['WaveDm'][:, gg],
                         'waveTp': stat_packet['Tp'][:, gg],
                         'waterLevel': stat_packet['WL'][:, gg],
                         'Umag': stat_packet['Umag'][:, gg],
                         'Udir': stat_packet['Udir'][:, gg ],
                         'Northing' : stat_packet['Northing'][0, gg],
                         'Easting': stat_packet['Easting'][0, gg],
                         'Latitude' : coords['Lat'],
                         'Longitude' : coords['Lon'],
                         'station_name': station,
                         'directionalWaveEnergyDensity': obse_packet['ncSpec'][:,gg,:,:],
                         'waveDirectionBins': obse_packet['ncDirs'],
                         'waveFrequency': obse_packet['Frequencies'],
                         'DX': dep_pack['DX'],
                         'DY': dep_pack['DY'],
                         'NI': dep_pack['NI'],
                         'NJ': dep_pack['NJ']}
            # move Local netCDF to Thredds
            TdsFldrBase = os.path.join(Thredds_Base,fldrArch)
            NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, station), datestring, model=model)
            NCname = 'CMTB-waveModels_{}_{}_{}_{}.nc'.format(model, version_prefix, station, datestring)
            outFileName = os.path.join(NCpath, NCname)  # Td

            if not os.path.exists(os.path.join(TdsFldrBase, station)):
                os.makedirs(os.path.join(TdsFldrBase, station)) # maameke the directory for th
            if not os.path.exists(os.path.join(TdsFldrBase, station, station + '.ncml')):
                inputOutput.makencml(os.path.join(TdsFldrBase, station, station+'.ncml'))
            assert os.path.isfile(globalyaml_fname_station), 'NetCDF yaml files are not created'
            makenc.makenc_Station(stat_data, globalyaml_fname=globalyaml_fname_station, flagfname=flagfname,
                                     ofname=outFileName, stat_yaml_fname=stat_yaml_fname)
    for gg, station in enumerate(NestedStations):
        stat_yaml_fname = station_var_yaml
        if station != ':': # stations marked with ':' for file names are not in the nested simulation
            # getting lat lon
            coords = gp.FRFcoord(stat_packet['Easting'][0, gg], stat_packet['Northing'][0, gg])
            stat_dataNest = {'time': nc.date2num(modelpacket_nest['time'][:], units='seconds since 1970-01-01 00:00:00'),
                         'waveHs': modelpacket_nest['Hs'][:, gg],
                         'waveTm': modelpacket_nest['Tm'][:, gg],
                         'waveDm': modelpacket_nest['WaveDm'][:, gg],
                         'waveTp': modelpacket_nest['Tp'][:, gg],
                         'waterLevel': modelpacket_nest['WL'][:, gg],
                         'Umag': modelpacket_nest['Umag'][:, gg],
                         'Udir': modelpacket_nest['Udir'][:, gg],
                         'Northing': modelpacket_nest['Northing'][0, gg],
                         'Easting': modelpacket_nest['Easting'][0, gg],
                         'Latitude': coords['Lat'],
                         'Longitude': coords['Lon'],
                         'station_name': station,
                         'directionalWaveEnergyDensity': obse_nested['ncSpec'][:, gg, :, :],
                         'waveDirectionBins': obse_nested['ncDirs'],
                         'waveFrequency': obse_nested['Frequencies'],
                         'DX': dep_nest['DX'],
                         'DY': dep_nest['DY'],
                         'NI': dep_nest['NI'],
                         'NJ': dep_nest['NJ']}
            # move Local netCDF to Thredds
            TdsFldrBase = os.path.join(Thredds_Base,fldrArch)
            NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, station), datestring, model=model)
            NCname = 'CMTB-waveModels_{}_{}_{}_{}.nc'.format(model, version_prefix, station, datestring)
            outFileName = os.path.join(NCpath, NCname)  # Td

            if not os.path.exists(os.path.join(TdsFldrBase, station)):
                os.makedirs(os.path.join(TdsFldrBase, station)) # maameke the directory for th
            if not os.path.exists(os.path.join(TdsFldrBase, station, station + '.ncml')):
                inputOutput.makencml(os.path.join(TdsFldrBase, station, station+'.ncml'))
            assert os.path.isfile(globalyaml_fname_station), 'NetCDF yaml files are not created'
            makenc.makenc_Station(stat_dataNest, globalyaml_fname=globalyaml_fname_station, flagfname=flagfname,
                                    ofname=outFileName, stat_yaml_fname=stat_yaml_fname)

    print "netCDF file's created for %s " %startTime
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ## COLLECT  ALL data (process, rotate, time pair and make comparison plots)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if plotFlag == True and np.size(stat_packet['time']) > 1:
        from testbedutils import waveLib as sbwave
        print '  Plotting Time Series Data '
        stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m']
        for gg, station in enumerate(stationList):
            print 'working on %s' %station
            # go get comparison data
            w = go.getWaveSpec(station)
            if 'time' in w:  # if there's data (not only location)
                if station in go.directional:
                    if full == False and station in go.directional:
                        w['dWED'], w['wavedirbin'] = sbwave.HPchop_spec(w['dWED'], w['wavedirbin'], angadj=angadj)
                    obsStats = sbwave.waveStat(w['dWED'], w['wavefreqbin'], w['wavedirbin'])
                else: # calc non directional stats
                    obsStats = sbwave.stats1D(w['fspec'], w['wavefreqbin'])
                if station in ['waverider-17m', 'awac-11m', 'waverider-26m']:
                    modStats = sbwave.waveStat(obse_packet['ncSpec'][:, gg, :, :], obse_packet['Frequencies'], obse_packet['ncDirs'])  # compute model stats here
                else:
                    modStats = sbwave.waveStat(obse_nested['ncSpec'][:, gg, :, :], obse_nested['Frequencies'], obse_nested['ncDirs'])  # compute model stats here
                    # time match data

                time, obsi, modi = sb.timeMatch(w['epochtime'],
                                                np.arange(w['time'].shape[0]),
                                                nc.date2num(stat_packet['time'][:], 'seconds since 1970-01-01'),
                                                np.arange(len(stat_packet['time']))) # time match
                # don't plot if theres only 1 dot on the plot... save time
                if station in go.directional:
                    plotList = ['Hm0', 'Tm', 'sprdF', 'sprdD', 'Tp', 'Dm']
                else:
                    plotList = ['Hm0', 'Tm', 'sprdF', 'Tp']
                for param in modStats:
                    if np.size(obsi) > 1 and np.size(modi) > 1 and param in plotList:
                        if param in ['Tm', 'Tp', 'Tm10', 'Tave']:
                            units = 's'
                            title = '%s period' % param
                        elif param in ['Hm0']:
                            units = 'm'
                            title = 'Wave Height %s ' %param
                        elif param in ['Dm', 'Dmp', 'Dp']:
                            units = 'degrees'
                            title = 'Direction %s' %param
                        elif param in ['sprdF']:
                            title = 'Spread %s' % param
                            units = 'Hz'
                        elif param in ['sprdD']:
                            units = 'degrees'
                            title = 'Spread %s ' % param

                        ofname = os.path.join(path_prefix, datestring, 'figures/CMTB-waveModels_{}_{}_station_{}_{}_{}.png'.format(model, version_prefix,
                                                                                                    station,
                                                                                                    param,
                                                                                                    datestring))
                        print 'plotting ' + ofname
                        dataDict = {'time': nc.num2date(time, 'seconds since 1970-01-01'),
                                    'obs': obsStats[param][obsi.astype(np.int)],
                                    'model': modStats[param][modi.astype(np.int)],
                                    'var_name': param,
                                    'units': units,
                                    'p_title': title}
                        oP.obs_V_mod_TS(ofname, dataDict, logo_path='ArchiveFolder/CHL_logo.png')
                        if station == 'waverider-26m' and param == 'Hm0':
                            # this is a fail safe to abort run if the boundary conditions don't
                            # meet quality standards below
                            bias = 0.1  # bias has to be within 10 centimeters
                            RMSE = 0.1  # RMSE has to be within 10 centimeters
                            if isinstance(dataDict['obs'], np.ma.masked_array) and ~dataDict['obs'].mask.any():
                                dataDict['obs'] = np.array(dataDict['obs'])
                            stats = sb.statsBryant(dataDict['obs'], dataDict['model'])
                            try:
                                assert stats['RMSE'] < RMSE, 'RMSE test on spectral boundary energy failed'
                                assert np.abs(stats['bias']) < bias, 'bias test on spectral boundary energy failed'
                            except:
                                print '!!!!!!!!!!FAILED BOUNDARY!!!!!!!!'
                                print 'deleting data from thredds!'
                                # os.remove(fieldOfname)
                                os.remove(outFileName)
                                raise RuntimeError('The Model Is not validating its offshore boundary condition')
