# -*- coding: utf-8 -*-
"""
  This code is the front and backend of the STWAVE module for the coastal model test bed, the scheme was the first developed
    and code has since been updated to reflect process updates, however some inefficiencies still exist.  please
    feel free to make these better and share your work on the CMTB repo
    Thanks,
    Spicer
"""
# import libraries
from testbedutils import anglesLib
from prepdata.inputOutput import stwaveIO
from plotting import operationalPlots as oP
from prepdata import inputOutput
from getdatatestbed import getDataFRF
import prepdata.prepDataLib as STPD
import datetime as DT
import glob, shutil, makenc, os, sys
import netCDF4 as nc
import numpy as np
from testbedutils import geoprocess as gp
from testbedutils import sblib as sb
from testbedutils import fileHandling
from subprocess import check_output

def STsimSetup(startTime, inputDict,allWind , allWL, allWave, bathy, loc_dict=None):
    """This Function is the master call for the  data preparation for the Coastal Model STWAVE runs.
    
    It is designed to be handed data then utilize prep_datalib for model pre-processing. All Files are labeled by
    the datestring.

    Args:
        startTime (str): input string in format YYYY-mm-ddThh:mm:ssZ
        inputDict (dict):  See documentation associated with input yamls
        allWind(dict): from get data with all wind from entire project set
        allWL(dict): from get data with all waterlevel from entire project set
        allWave(dict): from get data with all wave from entire project set
        gaugelocs(list): provides save points (lat/lon)
        
    Returns:
        nproc_parent (int): number of processors to run simultion for paret sim, will return -1 if this
            script things it should abort run
        nproc_nest (int): number of processors to run simulation in for nest, will return -1 is this script
            thinks it should abort run

    """
    # unpack dictionary -- Required
    version_prefix = inputDict['modelSettings']['version_prefix']
    timerun =  inputDict['simulationDuration']
    plotFlag = inputDict.get('plotFlag', True)
    background_grid_nested = inputDict['modelSettings']['gridDEP_nested']
    background_grid_parent = inputDict['modelSettings']['gridDEP_parent']
    path_prefix = inputDict['path_prefix']
    runNested = inputDict['modelSettings'].get('runNested', True)
    verbose = inputDict.get('verbose', True)
    # if 'ForcedSurveyDate' in inputDict['modelSettings']:
    #     ForcedSurveyDate = inputDict['modelSettings']['ForcedSurveyDate']
    #     FSDpieces = ForcedSurveyDate.split('-')
    #     ForcedSurveyDate = DT.datetime(int(FSDpieces[0]), int(FSDpieces[1]), int(FSDpieces[2]))
    # else:
    #     ForcedSurveyDate = None
    if not os.path.isfile(background_grid_parent):
        raise EnvironmentError('check your background Grid parent file name')
    if not os.path.isfile(background_grid_nested):
        raise EnvironmentError('Check your Background grid Nested file name')
    # parse all input data
    rawspec = allWave
    rawwind = allWind
    rawWL = allWL
    # _________________________________________________________________________________________________________________
    # defaults of the setup
    numNest = 3  # number of points to use as nesting seed

    # ___________________define version parameters_________________________________
    if version_prefix.lower() == 'fp':
        full = True # full plane
    elif version_prefix.lower() in ['hp', 'cbhp']:
        full = False  # half plane
    elif version_prefix.lower() in ['cb', 'cbthresh', 'cbt1', 'cbt2']:
        background_grid_nested = inputDict['gridDEP_nested'].replace('5', '10') # make this dummy proof
        assert startTime[10] == 'T', 'End time for simulation runs must be in the format YYYY-MM-DDThh:mm:ssZ'
        full = False # cbathy simulations run in half plane
    else:
        raise NameError('Need version prefix to run')

    # __________________set times _________________________________________________
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    dateString = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime

    # __________________Make Diretories_____________________________________________
    fileHandling.makeCMTBfileStructure(path_prefix, dateString)
    
    # __________________initalize needed classes ___________________________________
    prepdata = STPD.PrepDataTools()
    stio = inputOutput.stwaveIO('')  # initializing io here so grid text can be written out
    
    ###################################################################################################################
    #######################   Begin Gathering Data      ###############################################################
    ###################################################################################################################
    ## _____________WAVES____________________________
    if 'time' not in rawspec:
        print("\n++++\nThere's STILL No Wave data between %s and %s \n++++\n" % (d1, d2))
        return -1, -1 ## abort runs
    
    # rotate, time match, interpolate wave spectra
    waveTimeList = sb.createDateList(d1, d2-DT.timedelta(seconds=3600), DT.timedelta(seconds=3600)) # time match list
    wavepacket = prepdata.prep_spec(rawspec, version_prefix, datestr=dateString, plot=plotFlag, full=full,
                                    outputPath=path_prefix, waveTimeList=waveTimeList)
    print("number of wave records %d with %d interpolated points" % (np.shape(wavepacket['spec2d'])[0],
                                                                     sum(wavepacket['flag'])))
    # ____________ BATHY ______________________
    if runNested is True:
        ofnameDep = os.path.join(path_prefix, dateString, '{}nested.dep'.format(dateString))
        # warnings.warn('GetData bathy is in get model data!')
        gridNodesNested = prepdata.GetOriginalGridFromSTWAVE(background_grid_nested[:-4]+'.sim', background_grid_nested)
    
        if version_prefix.lower() in ['fp', 'hp', 'cbhp']:
            # get data first
            # bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate)
            # first find the nodes of the grid
            gridName='version_%s_SurveyDate_%s_SurveyNumber_%d' %(version_prefix, bathy['time'].strftime('%Y-%m-%d'),
                                                                  bathy['surveyNumber'])
    
        elif version_prefix.lower() == 'cb':
            # bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate, cBKF=True)
            gridName='version_{}_SurveyDate_{}'.format(version_prefix, bathy['time'].strftime('%Y-%m-%dT%H%M%SZ'))
    
        elif version_prefix.lower() == 'cbthresh':
            # bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate, cBKF_T=True)
            gridName='version_{}_SurveyDate_{}'.format(version_prefix, bathy['time'].strftime('%Y-%m-%dT%H%M%SZ'))
    
        print('Sim start: %s\nSim End: %s\nSim bathy chosen: %s' % (d1, d2, bathy['time']))
        # prep the grid to match the STWAVE domain in example grid file
        NestedBathy = prepdata.prep_Bathy(bathy, gridNodesNested, gridName=gridName, positiveDown=True)

    # _____________WINDS______________________
    if verbose: print('_________________\nprep wind data')
    # average and rotate winds
    windpacket = prepdata.prep_wind(rawwind, wavepacket['epochtime'], maxdeadrecord=6)

    # ___________WATER LEVEL__________________
    if verbose: print('_________________\nprep Water Level Data')
    # average WL
    WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])

    ## ___________CURRENTS_____________________
    # print '______________\nGetting Currents'
    curpacket = None

    # ##################################################################
    # check data do you have any problems
    print('Running Data Check\n+++++++++++++++++++++++++')
    prepdata.data_check(wavepacket, windpacket, WLpacket, curpacket)
    ##___________________________________________________________________________
    ##  Get sensor locations and add to sim file start
    sys.path.append('/home/spike/repos/TDSlocationGrabber')
    from frfTDSdataCrawler import query
    print('  TODO: handle TDS location grabber')
    dataLocations = query(d1, d2, inputName='/home/spike/repos/TDSlocationGrabber/database', type='waves')
    # # get gauge nodes x/y new idea: put gauges into input/output instance for the model, then we can save it
    statloc = []
    for _, gauge in enumerate(['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m',
                                'adop-3.5m', 'xp200m', 'xp150m', 'xp125m']):
        ii = np.argwhere(gauge == dataLocations['Sensor']).squeeze()
        coord = gp.FRFcoord(dataLocations['Lat'][ii], dataLocations['Lon'][ii],coordType='LL')
        print(' added: {} at xFRF {} yFRF {}'.format(gauge, coord['xFRF'], coord['yFRF']))
        statloc.append([coord['StateplaneE'], coord['StateplaneN']])
    statloc = np.array(statloc)

    #old way to grab gauge data
    # statloc =  []
    # for gauge in list(loc_dict.keys()):
    #     coords = loc_dict[gauge]
    #     try:
    #         statloc.append([coords['spE'], coords['spN']])
    #     except KeyError:
    #         continue
    # statloc = np.array(statloc)
    
    
    # assign nesting points in i/j grid coordinates
    if runNested is True:
        # go get the nesting points
        whichpointsy = np.linspace(gridNodesNested['yFRF'][0], gridNodesNested['yFRF'][-1], numNest)
        nestLocDict = {} #initalize nesting output locations
        for key in whichpointsy:
            nestLocDict[key]= {'xFRF': gridNodesNested['xFRF'].max(),'yFRF':key}
    else:
        nestLocDict = False
    ###################################################################################################################
    #######################   Begin Writing Files   ###################################################################
    ###################################################################################################################
    # Last thing to do ... write files file
    print('WRITING simulation Files')
    # now copy the outer domain to the local directory
    inputOutput.write_flags(dateString, path_prefix, wavepacket, windpacket, WLpacket, curpacket, gridFlag=False)
    shutil.copy2(background_grid_parent, os.path.join(path_prefix, dateString, '%s.dep' % dateString))
    stio.write_spec(dateString, path=path_prefix, STwavepacket=wavepacket)
    nproc_parent = stio.write_sim(dateString, path=path_prefix, snapbase=wavepacket['snapbase'], nested=0,
                                  windpacket=windpacket, WLpacket=WLpacket, curpacket=curpacket, statloc=statloc,
                                  full=full, nestpoints=nestLocDict)
    if runNested is True:
        stio.write_dep(ofnameDep, NestedBathy)
        nproc_nest = stio.write_sim(dateString, path=path_prefix, snapbase=wavepacket['snapbase'],
                                nested=1, windpacket=windpacket, WLpacket=WLpacket, curpacket=curpacket,
                                statloc=statloc, full=full)
    else:
        nproc_nest = None
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
    plotFlag = inputDict.get('plotFlag', True)
    model = inputDict.get('model', 'stwave')
    version_prefix = inputDict['modelSettings']['version_prefix']
    path_prefix = inputDict.get('path_prefix', "{}".format(version_prefix))
    simulationDuration = inputDict['simulationDuration']
    Thredds_Base = inputDict.get('netCDFdir', '/home/{}/thredds_data/'.format(check_output('whoami', shell=True)[:-1]))
    angadj = 70  # angle to adjust the
    runNested = inputDict['modelSettings'].get('runNested', True)

    ######### model static global setups
    if type(simulationDuration) == str:
        simulationDuration = int(simulationDuration)
    
    # __________________ define paths and times ___________________________________
    startTime = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    endTime = startTime + DT.timedelta(0, simulationDuration * 3600, 0)
    dateString = startTime.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, dateString)
    print(('\nBeggining of Analyze Script\nLooking for file in {}'.format(fpath)))
    print(('\nData Start: {}  Finish: {}'.format(startTime, endTime)))

    # __________________initalize needed classes ___________________________________
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

    stat_packet = stio.statload(nested=False)  # loads modeled data (parent station file)
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
    # creating template for spec to go into netCDF files
    obse_packet['ncSpec'] = np.ones((obse_packet['spec'].shape[0], obse_packet['spec'].shape[1],  obse_packet[
        'spec'].shape[2], 72)) * 1e-6
    
    if runNested:
        obse_nested = stio.obseload(nested=True)
        obse_nested['ncSpec'] = np.ones((obse_nested['spec'].shape[0], obse_nested['spec'].shape[1],  obse_nested[
            'spec'].shape[2], 72)) * 1e-6
        
    # rotating the spectra back to true north (TN) then
    for station in range(0, np.size(obse_packet['spec'], axis=1)):
        obse_packet['ncSpec'][:, station, :, :], obse_packet['ncDirs'] = prepdata.grid2geo_spec_rotate(
            obse_packet['directions'], obse_packet['spec'][:, station, :, :])
        # converting m^2/Hz/radians back to m^2/Hz/degree
        # note that units of degrees are on the denominator which requires a deg2rad conversion instead of rad2deg
        obse_packet['ncSpec'][:, station, :, :] = np.deg2rad(obse_packet['ncSpec'][:, station, :, :])

        if runNested:
            obse_nested['ncSpec'][:, station, :, :], obse_nested['ncDirs'] = prepdata.grid2geo_spec_rotate(
                                                      obse_nested['directions'], obse_nested['spec'][:, station, :, :])
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
    print('  ..begin loading spatial files ....')
    dep_pack = prepdata.GetOriginalGridFromSTWAVE(stio.simfname[0], stio.depfname[0])
    Tp_pack = stio.genLoad(nested=False, key='waveTp')
    # Tp_pack = stio.TPload(nested=0)     # this function is currently faster than genLoad (a generalized load function)
    wave_pack = stio.genLoad(nested=False, key='wave')
    if runNested:
        rad_nest = stio.genLoad('rad', nested=True)
        break_nest = stio.genLoad('break', nested=True)
        dep_nest = prepdata.GetOriginalGridFromSTWAVE(stio.simfname_nest[0], stio.depfname_nest[0])
        Tp_nest = stio.genLoad(key='waveTp', nested=True)
        # wave_pack2 = stio.genLoad('wave', nested=False)
        wave_nest = stio.genLoad(key='wave', nested=True)
    
    print('Files loaded, in %s ' %(DT.datetime.now() - d))

    if plotFlag == True:
        print("   BEGIN PLOTTING ")
        d = DT.datetime.now()
        plotFnameRegional = 'figures/CMTB_waveModels_STWAVE_{}_Regional-'.format(version_prefix)
        
        if runNested:
            plotFnameLocal = 'figures/CMTB_waveModels_STWAVE_{}_Local-'.format(version_prefix)
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
            oP.plotSpatialFieldData(dep_nest, Hs_nest_plot, os.path.join(fpath, plotFnameLocal + 'Hs'), nested=True,
                                    directions=wave_nest['Dm_field'])
            oP.plotSpatialFieldData(dep_nest, Tm_nest_plot, os.path.join(fpath, plotFnameLocal + 'Tm'), nested=True)
            # oP.plotSpatialFieldData(dep_nest, Dm_nest_plot, plotFnameLocal + 'Dm', fpath, nested=True)
            oP.plotSpatialFieldData(dep_nest, rads_nest_plot_y, os.path.join(fpath, plotFnameLocal + 'xRG'), nested=True)
            oP.plotSpatialFieldData(dep_nest, rads_nest_plot_x, os.path.join(fpath, plotFnameLocal + 'yRG'), nested=True)
            oP.plotSpatialFieldData(dep_nest, break_nest_plot, os.path.join(fpath, plotFnameLocal + 'break'), nested=True)

            # localDm = sorted(glob.glob(os.path.join(fpath,'figures/*Local-Dm*.png')))
            # sb.makegif(localDm, fpath+ plotFnameLocal + 'Dm_%s.gif' %(dateString))
            # [os.remove(ff) for ff in localDm]
            localTm = sorted(glob.glob(fpath + '/figures/*Local-Tm*.png'))
            localHs = sorted(glob.glob(fpath + '/figures/*Local-Hs*.png'))
            localTp = sorted(glob.glob(fpath + '/figures/*Local-Tp*.png'))
            localxHs = sorted(glob.glob(fpath + '/figures/*LocalxShoreWaveHeight*.png'))
            localxRad = sorted(glob.glob(fpath + '/figures/*Local-xRG*.png'))
            localyRad = sorted(glob.glob(fpath + '/figures/*Local-yRG*.png'))
            localBreak = sorted(glob.glob(fpath + '/figures/*Local-break*.png'))
            sb.makegif(localTm, os.path.join(fpath, plotFnameLocal + 'Tm_%s.gif'%(dateString)))
            sb.makegif(localHs, os.path.join(fpath, plotFnameLocal + 'Hs_%s.gif' %(dateString)))
            sb.makegif(localTp, os.path.join(fpath, plotFnameLocal + 'Tp_%s.gif' %(dateString)))
            sb.makegif(localxHs, os.path.join(fpath, plotFnameLocal + 'xShoreWaveHeight_%s.gif' %(dateString)))
            sb.makegif(localxRad, os.path.join(fpath, plotFnameLocal + 'xRadGrad_{}.gif'.format(dateString)))
            sb.makegif(localyRad, os.path.join( fpath, plotFnameLocal + 'yRadGrad_{}.gif'.format(dateString)))
            sb.makegif(localBreak, os.path.join(fpath, plotFnameLocal + 'Break_{}.gif'.format(dateString)))
            [os.remove(ff) for ff in localHs]
            [os.remove(ff) for ff in localTp]
            [os.remove(ff) for ff in localxHs]
            [os.remove(ff) for ff in localxRad]
            [os.remove(ff) for ff in localyRad]
            [os.remove(ff) for ff in localBreak]
            [os.remove(ff) for ff in localTm]

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

        oP.plotSpatialFieldData(dep_pack, dep_parent_plot, prefix=os.path.join(fpath, plotFnameRegional + 'bathy'),
                                namebase=dateString, nested=False)
        oP.plotSpatialFieldData(dep_pack, Tm_parent_plot, prefix=os.path.join(fpath, plotFnameRegional + 'Tm'),
                                namebase=dateString, nested=False)
        oP.plotSpatialFieldData(dep_pack, Hs_parent_plot, prefix=os.path.join(fpath, plotFnameRegional + 'Hs'),
                                namebase=dateString, nested=False, directions=wave_pack['Dm_field'])
        oP.plotSpatialFieldData(dep_pack, Tp_parent_plot, prefix=os.path.join(fpath, plotFnameRegional + 'Tp'),
                                namebase=dateString, nested=False)

        # ################################
        # Make GIFs from Images          #
        # ## #############################


        # regional
        regTm = sorted(glob.glob(fpath + '/figures/*Regional-Tm*.png'))
        regHs = sorted(glob.glob(fpath + '/figures/*Regional-Hs*.png'))
        regTp = sorted(glob.glob(fpath + '/figures/*Regional-Tp*.png'))
        sb.makegif(regTm, os.path.join(fpath, plotFnameRegional +  'Tm_%s.gif' %(dateString)))
        sb.makegif(regHs, os.path.join(fpath, plotFnameRegional + 'Hs_%s.gif' %(dateString)))
        sb.makegif(regTp, os.path.join(fpath, plotFnameRegional + 'Tp_%s.gif' %(dateString)))
        [os.remove(ff) for ff in regTm]
        [os.remove(ff) for ff in regHs]
        [os.remove(ff) for ff in regTp]

        print('-- Spatial plots were made in %s ' %(DT.datetime.now() - d))

    # ################################
    # Make NETCDF files              #
    # ## #############################
    #  local grid global metadata
    # Regional grid Global metadata
    flagfname = fpath + '/Flags%s.out.txt' % dateString  # startTime # the name of flag file
    if np.median(dep_pack['bathy']) < 0:
        dep_pack['bathy'] = - dep_pack['bathy']

    if dep_pack['gridFname'].lower().strip('"')=='regional_17mgrid_50m':
        # check domain on server - assume that it's correct
        wave_pack, dep_pack, Tp_pack = prepdata.Fixup17mGrid(version_prefix, wave_pack, dep_pack, Tp_pack)
    
    # writing data libraries
    regionalDataLib = {
        'time': nc.date2num(wave_pack['time'], units='seconds since 1970-01-01 00:00:00'),
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
        'NJ': dep_pack['NJ']
        }

    cmtbLocalFldrArch = os.path.join(model, version_prefix)
    varYml = 'yaml_files/waveModels/{}/Field_var.yml'.format(model)
    regGlobYml = 'yaml_files/waveModels/{}/{}/Field_Regional_{}_globalmeta.yml'.format(model, version_prefix,
                                                                                       version_prefix)
    outFileName = fileHandling.makeTDSfileStructure(Thredds_Base, cmtbLocalFldrArch, dateString, 'Regional-Field')
    assert os.path.isfile(regGlobYml), 'NetCDF yaml files are not created'
    makenc.makenc_field(data_lib=regionalDataLib, globalyaml_fname=regGlobYml, flagfname=flagfname,
                        ofname=outFileName, var_yaml_fname=varYml)
    ######################################################################################################################
    ######################################################################################################################
    ##################################  Wave Station Files HERE (loop) ###################################################
    ######################################################################################################################
    ######################################################################################################################

    if runNested is True:
        if np.median(dep_nest['bathy']) < 0:
            dep_nest['bathy'] = -dep_nest['bathy']
        # Making record of the Date of the survey/inversion in datetime format
        gridNameSplit = dep_nest['gridFname'].split('_')
        if 'SurveyDate' in gridNameSplit:
            bathyDateA = gridNameSplit[np.argwhere(np.array(gridNameSplit[:]) == 'SurveyDate').squeeze() + 1].strip('"')
            # making the bathymetry grid
            try:
                bathyTime = DT.datetime.strptime(bathyDateA, '%Y-%m-%dT%H%M%SZ')
            except ValueError:
                bathyTime = DT.datetime.strptime(bathyDateA, '%Y-%m-%d')
    
        localDataLib = {
            'time': nc.date2num(wave_nest['time'], units='seconds since 1970-01-01 00:00:00'),
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
            'DY': dep_nest['DY']
            }
        locGlobYml = 'yaml_files/waveModels/{}/{}/Field_Local_{}_globalmeta.yml'.format(model, version_prefix, version_prefix)
        outFileName = fileHandling.makeTDSfileStructure(Thredds_Base, cmtbLocalFldrArch, dateString, 'Local-Field')
        assert os.path.isfile(regGlobYml), 'NetCDF yaml files are not created'
        makenc.makenc_field(data_lib=localDataLib, globalyaml_fname=locGlobYml, flagfname=flagfname,
                            ofname=outFileName, var_yaml_fname=varYml)
        NestedStations = [':', ':', ':','8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m', ]
        RegionalStations = ['waverider-26m', 'waverider-17m', 'awac-11m', ':', ':', ':', ':', ':', ':', ':'] # where :'s are for sims in the nested

        if (version_prefix == 'CB' or version_prefix == 'CBHP') and \
            (startTime >= DT.datetime(2015,10,15) and endTime < DT.datetime(2015, 11, 1)):
            NestedStations.extend(('Station_p11', 'Station_p12', 'Station_p13', 'Station_p14', 'Station_p21', 'Station_p22',
                                   'Station_p23', 'Station_p24'))
    else: # running only regional grid
        RegionalStations = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m',
                            'adop-3.5m', 'xp200m', 'xp150m', 'xp125m'] # where :'s are for sims in the nested
    # writing station files from regional/parent simulation
    for gg, station in enumerate(RegionalStations):
        if station != ':':
            # getting lat lon
            coords = gp.FRFcoord(stat_packet['Easting'][0, gg], stat_packet['Northing'][0, gg])
            stat_data = {
                'time': nc.date2num(stat_packet['time'][:], units='seconds since 1970-01-01 00:00:00'),
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
                'NJ': dep_pack['NJ']
                }
        
            globalyaml_fname_station = 'yaml_files/waveModels/{}/{}/Station_{}_globalmeta.yml'.format(model,
                                                                                      version_prefix, version_prefix)
            station_var_yaml = 'yaml_files/waveModels/{}/Station_Directional_Wave_var.yml'.format(model)
            outFileName = fileHandling.makeTDSfileStructure(Thredds_Base, cmtbLocalFldrArch, dateString, 'Field')
            makenc.makenc_Station(stat_data, globalyaml_fname=globalyaml_fname_station, flagfname=flagfname,
                                  ofname=outFileName, stat_yaml_fname=station_var_yaml)

    

    print("netCDF file's created for {} ".format(startTime))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    ## COLLECT  ALL data (process, rotate, time pair and make comparison plots)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    go=getDataFRF.getObs(startTime, endTime)
    if plotFlag == True and np.size(stat_packet['time']) > 1:
        from testbedutils import waveLib as sbwave
        print('  Plotting Time Series Data ')
        stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m']
        for gg, station in enumerate(stationList):
            print('working on %s' %station)
            # go get comparison data
            w = go.getWaveSpec(station)
            if w is not None and 'time' in w:  # if there's data (not only location)
                if station in go.directionalWaveGaugeList:
                    if full == False and station in go.directionalWaveGaugeList:
                        w['dWED'], w['wavedirbin'] = sbwave.HPchop_spec(w['dWED'], w['wavedirbin'], angadj=angadj)
                    obsStats = sbwave.waveStat(w['dWED'], w['wavefreqbin'], w['wavedirbin'])
                else: # calc non directionalWaveGaugeList stats
                    obsStats = sbwave.stats1D(w['fspec'], w['wavefreqbin'])
                
                if station in ['waverider-17m', 'awac-11m', 'waverider-26m']:
                    modStats = sbwave.waveStat(obse_packet['ncSpec'][:, gg, :, :], obse_packet['Frequencies'],
                                               obse_packet['ncDirs'])  # compute model stats here
                else:
                    if runNested is True:
                        modStats = sbwave.waveStat(obse_nested['ncSpec'][:, gg, :, :], obse_nested['Frequencies'],
                                               obse_nested['ncDirs'])  # compute model stats here

                # time match data
                time, obsi, modi = sb.timeMatch(w['epochtime'],
                                                np.arange(w['time'].shape[0]),
                                                nc.date2num(stat_packet['time'][:], 'seconds since 1970-01-01'),
                                                np.arange(len(stat_packet['time']))) # time match
                # don't plot if theres only 1 dot on the plot... save time
                if station in go.directionalWaveGaugeList:
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

                        ofname = os.path.join(path_prefix, dateString,
                                  'figures/CMTB-waveModels_{}_{}_station_{}_{}_{}.png'.format(model, version_prefix,
                                                                                            station, param, dateString))
                        print('plotting ' + ofname)
                        dataDict = {'time': nc.num2date(time, 'seconds since 1970-01-01',
                                                        only_use_cftime_datetimes=False),
                                    'obs': obsStats[param][obsi.astype(np.int)],
                                    'model': modStats[param][modi.astype(np.int)],
                                    'var_name': param,
                                    'units': units,
                                    'p_title': title}
                        oP.obs_V_mod_TS(ofname, dataDict, logo_path='ArchiveFolder/CHL_logo.png')
                        if station == 'waverider-26m' and param == 'Hm0':
                            print('   skipping boundary spectral comparison')
                            continue
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
                                print('!!!!!!!!!!FAILED BOUNDARY!!!!!!!!')
                                print('deleting data from thredds!')
                                # os.remove(fieldOfname)
                                os.remove(outFileName)
                                raise RuntimeError('The Model Is not validating its offshore boundary condition')
    # writing data out for UQ effort. Starting with parent domain (can complicate further as necessary)
    outData = regionalDataLib
    return outData