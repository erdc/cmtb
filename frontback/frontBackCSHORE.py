from prepdata import inputOutput
import os
import datetime as DT
import numpy as np
from getdatatestbed.getDataFRF import getObs, getDataTestBed
from testbedutils.sblib import timeMatch, timeMatch_altimeter, makeNCdir
import plotting.operationalPlots as oP
import makenc
from subprocess import check_output
import prepdata.prepDataLib as PDL

def CSHOREsimSetup(startTime, inputDict):
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
    # pull the stuff I need out of the dict
    timerun = inputDict['simulationDuration']
    version_prefix = inputDict['version_prefix']
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
    start_time = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    bathy_loc_List = np.array(['integrated_bathy', 'survey'])

    assert start_time.minute == 0 and start_time.second == 0 and start_time.microsecond == 0, 'Your simulation must start on the hour!'

    end_time = start_time + DT.timedelta(days=0, hours=timerun) # removed for ilab=1 , minutes=1)
    date_str = start_time.strftime('%Y-%m-%dT%H%M%SZ')
    # start making my metadata dict
    meta_dict = {'startTime': startTime,
                 'timerun': timerun,
                 'time_step': time_step,
                 'dx': dx,
                 'fric_fac': fric_fac,
                 'version': version_prefix}
    ftime = timerun * 3600  # [sec] final time, dictates model duration
    dt = time_step * 3600  # time interval (sec) for wave and water level conditions
    BC_dict = {'timebc_wave': np.arange(0, ftime + dt, dt)}

    # ______________________________________________________________________________
    # __________________Make Diretories_____________________________________________
    if not os.path.exists(path_prefix + date_str):  # if it doesn't exist
        os.makedirs(path_prefix + date_str)  # make the directory
    if not os.path.exists(path_prefix + date_str + "/figures/"):
        os.makedirs(path_prefix + date_str + "/figures/")

    print("Model Time Start : %s  Model Time End:  %s" % (start_time, end_time))
    print("Files will be placed in {0} folder".format(path_prefix + date_str))

    # ______________________________________________________________________________
    # _____________Initialize the Classes___________________________________________
    prep = PDL.PrepDataTools()
    # this is the getObs instance for waves!!!
    # it includes three hours to either side so the simulation can still run if we are missing wave data at the start and end points!!!
    frf_DataW = getObs(start_time - DT.timedelta(days=0, hours=3, minutes=0), end_time + DT.timedelta(days=0, hours=3, minutes=1), server)
    # go ahead and pull both wave gauges so I won't have to repeat this line over and over!
    print("_________________\nGathering Wave Data")
    wave_data8m = frf_DataW.getWaveSpec(gaugenumber=12)
    wave_data6m = frf_DataW.getWaveSpec(gaugenumber=4)
    # getObs instance for bathy!! - only pull something that is on that day!
    frf_DataB = getObs(start_time, end_time + DT.timedelta(days=0, hours=0, minutes=1), server)
    cmtb_data = getDataTestBed(start_time, end_time + DT.timedelta(days=0, hours=0, minutes=1), server)
    cshore_io_O = inputOutput.cshoreIO()  # initalize input output

    # _____________________________________________________________________________
    # ______________Decision Time - Fixed vs. Mobile_______________________________
    if version_prefix == 'FIXED':
        # if it is fixed, first thing to do is get waves

        ## _____________WAVES____________________________
        o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time)
        assert o_dict is not None, 'Simulation broken.  Missing wave data!'

        # ____________ BATHY ______________________
        print('\n____________________\nGetting Bathymetric Data\n')

        # what do I use as my initial bathy?
        assert bathy_loc in bathy_loc_List, "Please enter a valid source bathymetry location \n Assigned location = %s must be in List %s" % (bathy_loc, bathy_loc_List)
        b_dict = {}
        if bathy_loc == 'survey':
            # is this profile number in the survey?
            prof_nums = frf_DataB.getBathyTransectProfNum()
            assert profile_num in prof_nums, 'Please begin simulations with a survey that includes profile number %s.' %(str(profile_num))
            # get the bathy packet
            bathy_data = frf_DataB.getBathyTransectFromNC(profilenumbers=profile_num)
        elif bathy_loc == 'integrated_bathy':
            # pull the bathymetry from the integrated product - see getdatatestbed module
            bathy_data = cmtb_data.getBathyIntegratedTransect()
        else:
            bathy_data = None
        # prep the packet
        b_dict = prep.prep_CSHOREbathy(bathy_data, bathy_loc, profile_num, dx, o_dict, fric_fac)

    elif version_prefix == 'MOBILE':

        # try to get it from prior cshore run!!!!
        try:
            Time_O = (DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') - DT.timedelta(days=1)).strftime('%Y-%m-%dT%H%M%SZ')
            # initialize the class
            params0, bc0, veg0, hydro0, sed0, morpho0, meta0 = cshore_io_O.load_CSHORE_results(path_prefix + Time_O)
            prev_wg = meta0['BC_gage']

            # which gage was it?
            o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time)
            assert o_dict is not None, 'Simulation broken.  Missing wave data!'

            # bathy stuff here
            bathy_data = {}
            bathy_data['meta0'] = meta0
            bathy_data['morpho0'] = morpho0
            b_dict = prep.prep_CSHOREbathy(bathy_data, 'prior_model', profile_num, dx, o_dict, fric_fac)

            # check to see if we stepped down and I have to adjust my x, zb
            if prev_wg == wave_data8m['name'] and meta_dict['BC_gage'] == wave_data6m['name']:
                b_dict = prep.prep_CSHOREbathy(b_dict, o_dict)

        except:
            # couldnt find old simulation - this is the start of a new simulation - proceed as normal

            # which gage was it?
            o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time)
            assert o_dict is not None, 'Simulation broken.  Missing wave data!'

            # ____________ BATHY ______________________
            print('\n____________________\nGetting Bathymetric Data\n')

            # what do I use as my initial bathy?
            bathy_loc_List = np.array(['integrated_bathy', 'survey'])
            assert bathy_loc in bathy_loc_List, "Please enter a valid source bathymetry location \n Assigned location = %s must be in List %s" % (
            bathy_loc, bathy_loc_List)
            b_dict = {}
            if bathy_loc == 'survey':
                # is this profile number in the survey?
                prof_nums = frf_DataB.getBathyTransectProfNum()
                assert profile_num in prof_nums, 'Please begin simulations with a survey that includes profile number %s.' % (
                    str(profile_num))
                # get the bathy packet
                bathy_data = frf_DataB.getBathyTransectFromNC(profilenumbers=profile_num)
            elif bathy_loc == 'integrated_bathy':
                # pull the bathymetry from the integrated product - see getdatatestbed module
                cmtb_data = getDataTestBed(start_time, end_time + DT.timedelta(days=0, hours=0, minutes=1), server)
                bathy_data = cmtb_data.getBathyIntegratedTransect()
            else:
                bathy_data = None
            # prep the packet
            b_dict = prep.prep_CSHOREbathy(bathy_data, bathy_loc, profile_num, dx, o_dict, fric_fac)

    # version 2.0 of MOBILE_RESET goes here...
    elif version_prefix == 'MOBILE_RESET':

        # first thing to check... am I resetting today
        if inputDict['reset']:
            # which wave gauge?
            o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time)
            assert o_dict is not None, 'Simulation broken.  Missing wave data!'

            # do I want the integrated bathy or the survey
            # ____________ BATHY ______________________
            print('\n____________________\nGetting Bathymetric Data\n')
            if bathy_loc == 'survey':
                # is this profile number in the survey?
                prof_nums = frf_DataB.getBathyTransectProfNum()
                assert profile_num in prof_nums, 'Please begin simulations with a survey that includes profile number %s.' % (str(profile_num))
                # get the bathy packet
                bathy_data = frf_DataB.getBathyTransectFromNC(profilenumbers=profile_num)
            elif bathy_loc == 'integrated_bathy':
                # pull the bathymetry from the integrated product - see getdatatestbed module
                bathy_data = cmtb_data.getBathyIntegratedTransect()
            else:
                bathy_data = None
            # prep the packet
            b_dict = prep.prep_CSHOREbathy(bathy_data, bathy_loc, profile_num, dx, o_dict, fric_fac)

        else:
            # no i am not
            b_dict = {}
            try:
                Time_O = (DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') - DT.timedelta(days=1)).strftime('%Y-%m-%dT%H%M%SZ')
                # initialize the class
                params0, bc0, veg0, hydro0, sed0, morpho0, meta0 = cshore_io_O.load_CSHORE_results(path_prefix + Time_O)
                prev_wg = meta0['BC_gage']

                # which gage was it?
                o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time, prev_wg=prev_wg)
                assert o_dict is not None, 'Simulation broken.  Missing wave data!'

                # bathy stuff here
                # ____________ BATHY ______________________
                print('\n____________________\nGetting Bathymetric Data\n')
                bathy_data = {}
                bathy_data['meta0'] = meta0
                bathy_data['morpho0'] = morpho0
                b_dict = prep.prep_CSHOREbathy(bathy_data, 'prior_model', profile_num, dx, o_dict, fric_fac)

                # check to see if we stepped down and I have to adjust my x, zb
                if prev_wg == wave_data8m['name'] and o_dict['BC_gage'] == wave_data6m['name']:
                    b_dict = prep.prep_CSHOREbathy(b_dict, o_dict)
            except:
                raise EnvironmentError('Simulation created an Error and can no longer proceed')

    # now assign my boundary condition stuff
    meta_dict['bathy_surv_num'] = b_dict['bathy_surv_num']
    meta_dict['bathy_surv_stime'] = b_dict['bathy_surv_stime']
    meta_dict['bathy_surv_etime'] = b_dict['bathy_surv_etime']
    meta_dict['bathy_prof_num'] = b_dict['bathy_prof_num']
    meta_dict['bathy_y_max_diff'] = b_dict['bathy_y_max_diff']
    meta_dict['bathy_y_sdev'] = b_dict['bathy_y_sdev']
    meta_dict['BC_FRF_X'] = b_dict['BC_FRF_X']
    meta_dict['BC_FRF_Y'] = b_dict['BC_FRF_Y']
    BC_dict['x'] = b_dict['x']
    BC_dict['zb'] = b_dict['zb']
    BC_dict['fw'] = b_dict['fw']

    # now assign my wave stuff?
    meta_dict['BC_gage'] = o_dict['BC_gage']
    meta_dict['blank_wave_data'] = o_dict['blank_wave_data']
    BC_dict['Hs'] = o_dict['Hs']
    BC_dict['Tp'] = o_dict['Tp']
    BC_dict['angle'] = o_dict['angle']

    # make sure wave data are appropriate
    assert 'Hs' in list(BC_dict.keys()), 'Simulation broken.  Wave data are missing for both 8m array and 6m AWAC!'

    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    try:
        # Pull water level data
        wl_data = prep.prep_WL(frf_DataW.getWL(), [start_time + DT.timedelta(days=0, seconds=tt) for tt in BC_dict['timebc_wave']])
        BC_dict['swlbc'] = wl_data['avgWL'] #gives me the avg water level at "date_list"
        BC_dict['Wsetup'] = np.zeros(len(BC_dict['timebc_wave']))  # we are HARD CODING the wave setup to always be zero!!!
        meta_dict['blank_wl_data'] = wl_data['flag']
        print('number of WL records %d, with %d interpolated points' % (np.size(wl_data['time']), sum(wl_data['flag'])))
    except (RuntimeError, TypeError):
        wl_data = None
    assert wl_data is not None, 'Error: water level data missing for simulation'

    # ___________TEMP AND SALINITY ______________________
    ctd_data = frf_DataB.getCTD()
    if ctd_data == None:
        BC_dict['salin'] = 30  # salin in ppt
        BC_dict['temp'] = 15  # water temp in degrees C
    else:
        # DLY note 11/06/2018: this may break if we ever get getCTD() to work and the time stamps are NOT
        # the same as the cshore time steps!
        BC_dict['salin'] = ctd_data['salin']  # salin in ppt
        BC_dict['temp'] = ctd_data['temp']  # water temp in degrees C

    # Last thing to do ... write files
    print('WRITING simulation Files')
    cshore_io = inputOutput.cshoreIO()

    # since we are changing ilab to be 1, I need to pare down my input stuff
    BC_dict['angle'] = BC_dict['angle'][0:-1]
    BC_dict['timebc_wave'] = BC_dict['timebc_wave'][0:-1]
    BC_dict['Hs'] = BC_dict['Hs'][0:-1]
    BC_dict['Wsetup'] = BC_dict['Wsetup'][0:-1]
    BC_dict['Tp'] = BC_dict['Tp'][0:-1]
    BC_dict['swlbc'] = BC_dict['swlbc'][0:-1]

    # write infile
    cshore_io.make_CSHORE_infile(path_prefix + date_str + '/infile', BC_dict, meta_dict)
    # write metadata file
    # cshore_io.write_flags(date_str, path_prefix, wavepacket, windpacket, WLpacket, curpacket, gridFlag)

def CSHORE_analysis(startTime, inputDict):
    """
    Args:
        startTime (str): this is the time that all the CSHORE runs are tagged by (e.g., '2012-12-31T00:30:30Z')
        inputDicts (dict): dictionary input
            version_prefix - right now we have MOBILE, MOBILE_RESET. FIXED
            workingDir - path to the working directory the user wants
            pFlag - do you want plots or not?
            netCDFdir - directory where the netCDF files will be saved
    Returns:
          None

    """
    version_prefix = inputDict['version_prefix']
    workingDir = inputDict['workingDirectory']
    pFlag = inputDict['pFlag']
    if 'netCDFdir' in list(inputDict.keys()):
        netCDFdir = inputDict['netCDFdir']
    else:
        whoami = check_output('whoami')[:-1]
        netCDFdir = 'home/%s/thredds_data' % whoami
    if 'THREDDS' in inputDict:
        server = inputDict['THREDDS']
    else:
        print('Chosing CHL thredds by Default, this may be slower!')
        server = 'CHL'
    model='CSHORE'
    #initialize the class
    cshore_io = inputOutput.cshoreIO()
    prepData = PDL.PrepDataTools()

    # get into the directory I need
    start_dir = workingDir
    path_prefix = os.path.join(model, version_prefix)  # data super directiory
    d_s = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    date_str = d_s.strftime('%Y-%m-%dT%H%M%SZ') # THE COLONS!!! startTime has COLONS!!!
    figurePath = os.path.join(start_dir, path_prefix, date_str, 'figures/')
    params, bc, veg, hydro, sed, morpho, meta = cshore_io.load_CSHORE_results(os.path.join(start_dir, path_prefix, date_str))
    # params - metadata about the run
    # bc - boundary condition data, but for some reason it does not include the initial conditions?
    # veg - vegetation information
    # hydro - current and wave information
    # sed - sediment information
    # morpho - bed elevation information

    test = np.append(bc['time_offshore'], max(bc['time_offshore']) + (bc['time_offshore'][1] - bc['time_offshore'][0]))
    times = np.array([d_s + DT.timedelta(seconds=s) for s in test])

    # change my coordinate system back to FRF!!!!!!!
    x_n = meta["BC_FRF_X"] - morpho['x'][0]
    model_time = times[-1]
    endTime = times[-1]
    startTime = times[0]
    go = getObs(d1=startTime, d2=endTime)
    # convert model time to epoch?
    # timeunits = 'seconds since 1970-01-01 00:00:00'
    # model_timeN = nc.date2num(model_time, timeunits)

    ###################################################################################################################
    ############################ Make NetCDF files ####################################################################
    ###################################################################################################################

    # data as loaded by the load cshore results function
    nc_dict = PDL.postProcessCshoreOutput(d_s=startTime, hydro=hydro, sed=sed, morpho=morpho, meta=meta)

    yaml_dir = os.path.dirname(os.path.abspath(__file__))
    # need to go up one
    yaml_dir = os.path.dirname(yaml_dir)

    if 'MOBILE' in version_prefix:
        globalYaml = os.path.join('yaml_files/{}/CSHORE_mobile_global.yml'.format(model))
        varYaml = os.path.join('yaml_files/{}/CSHORE_mobile_var.yml'.format(model))
    elif 'FIXED' in version_prefix:
        globalYaml = os.path.join('yaml_files/{}/CSHORE_fixed_global.yml'.format(model))
        varYaml = os.path.join('yaml_files/{}/CSHORE_fixed_var.yml'.format(model))
    else:
        raise NotImplementedError('please check version prefix')

    assert globalYaml is not None, 'CSHORE_analysis Error: Version prefix not recognized'

    NCpath = makeNCdir(netCDFdir, version_prefix, date_str, model='CSHORE')

    # make the name of this nc file your OWN SELF BUM!
    NCname = 'CMTB-morphModels_{}_{}_{}.nc'.format(model, version_prefix, date_str)
    makenc.makenc_CSHORErun(os.path.join(NCpath, NCname), nc_dict, globalYaml, varYaml)

    ###################################################################################################################
    ############################ plotting #############################################################################
    ###################################################################################################################
    # make the plots
    if pFlag:

        #########################################################################
        ## first get hydro data (these data are used for both fixed and mobile bed simulations
        ################################################################################################################
        # gather wave, current, data. each of these functions gets waves and current data and repackages into one dictionary
        # This is probably not the best way to handle this, or it may be in the wrong place ... moving forward for now
        obs_dict_waves, obs_dict_currents = {}, {}

        Adopp_35 = oP.wave_PlotData('adop-3.5m', model_time, times)
        AWAC6m = oP.wave_PlotData('awac-6m', model_time, times)
        lidar = oP.lidar_PlotData(times)
        obs_dict = {'Adopp_35': Adopp_35,
                    'AWAC6m': AWAC6m,
                    'lidar': lidar, }

        if 'mobile'.lower() in version_prefix.lower():
                ########################################################################################################
                ## then do morph 'only' evaluations --- these are used later for summary plots if they're available
                ########################################################################################################
                # Altimeter data
                Alt05 = oP.alt_PlotData('Alt05', model_time, times)
                Alt04 = oP.alt_PlotData('Alt04', model_time, times)
                Alt03 = oP.alt_PlotData('Alt03', model_time, times)
                obs_dict['Alt05'] = Alt05  # set these for summary plots later
                obs_dict['Alt04'] = Alt04
                obs_dict['Alt03'] = Alt03
                # go ahead and time match model to the altimeter data
                if Alt05['TS_toggle']:
                    # ALT05
                    obs_loc = round(Alt05['xFRF'])
                    mod_zb = morpho['zb'][:,
                             np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    Alt05['time'], Alt05['zb'], mod_n = timeMatch_altimeter(Alt05['time'], Alt05['zb'], times[1:],
                                                                            mod_zb)
                    Alt05['plot_ind'] = np.where(
                        abs(Alt05['time'] - model_time) == min(abs(Alt05['time'] - model_time)), 1, 0)
                    obs_loc = round(Alt05['xFRF'])
                    mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    matchObs = prepData.prep_obs2mod(Alt05['time'], Alt05['zb'], times[1:])

                    # check to see if we masked anything!!
                    if np.sum(matchObs['mask']) > 0:
                        mod_zb = mod_zb[np.where(~matchObs['mask'])]

                    p_dict = {'time': matchObs['time'],
                              'obs': matchObs['meanObs'],
                              'model': mod_zb,
                              'var_name': 'Bottom Elevation',
                              'units': 'm',
                              'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'Alt-05')}

                    if np.size(p_dict['obs']) >= 2:
                        oP.obs_V_mod_TS(figurePath + 'alt05_BE.png', p_dict)


                if Alt04['TS_toggle']:
                    obs_loc = round(Alt04['xFRF'])
                    mod_zb = morpho['zb'][:,
                             np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    Alt04['time'], Alt04['zb'], mod_n = timeMatch_altimeter(Alt04['time'], Alt04['zb'], times[1:],
                                                                            mod_zb)
                    Alt04['plot_ind'] = np.where(abs(Alt04['time'] - model_time) == min(abs(Alt04['time'] - model_time)), 1, 0)
                    obs_loc = round(Alt04['xFRF'])
                    mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    matchObs = prepData.prep_obs2mod(Alt04['time'], Alt04['zb'], times[1:])

                    # check to see if we masked anything!!
                    if np.sum(matchObs['mask']) > 0:
                        mod_zb = mod_zb[np.where(~matchObs['mask'])]

                    p_dict = {'time': matchObs['time'],
                              'obs': matchObs['meanObs'],
                              'model': mod_zb,
                              'var_name': 'Bottom Elevation',
                              'units': 'm',
                              'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'Alt-04')}

                    if np.size(p_dict['obs']) >= 2:
                        oP.obs_V_mod_TS(figurePath + 'alt04_BE.png', p_dict)


                if Alt03['TS_toggle']:
                    obs_loc = round(Alt03['xFRF'])
                    mod_zb = morpho['zb'][:,
                             np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    Alt03['time'], Alt03['zb'], mod_n = timeMatch_altimeter(Alt03['time'], Alt03['zb'], times[1:],
                                                                            mod_zb)
                    Alt03['plot_ind'] = np.where(
                        abs(Alt03['time'] - model_time) == min(abs(Alt03['time'] - model_time)), 1, 0)
                    obs_loc = round(Alt03['xFRF'])
                    mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    matchObs = prepData.prep_obs2mod(Alt03['time'], Alt03['zb'], times[1:])

                    # check to see if we masked anything!!
                    if np.sum(matchObs['mask']) > 0:
                        mod_zb = mod_zb[np.where(~matchObs['mask'])]

                    p_dict = {'time': matchObs['time'],
                              'obs': matchObs['meanObs'],
                              'model': mod_zb,
                              'var_name': 'Bottom Elevation',
                              'units': 'm',
                              'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'Alt-03')}

                    if np.size(p_dict['obs']) >= 2:
                        oP.obs_V_mod_TS(figurePath + 'alt03_BE.png', p_dict)

        #######################################
        # plot hydro (waves and currents) not runup
        #######################################
        go.waveGaugeList.extend(['lidarWaveGauge080', 'lidarWaveGauge090', 'lidarWaveGauge100',
                              'lidarWaveGauge110', 'lidarWaveGauge140'])
        plotVarsWaves = ['Hs']
        plotVarsCurrents = ['U', 'V']
        # first plot through waves
        for gauge in go.waveGaugeList:
            print('Gathering Data for wave comparisons on {}'.format(gauge))
            obs_dict_waves[gauge] = go.getWaveSpec(gauge)
            if 'time' in obs_dict_waves[gauge].keys() and  x_n.max() >=obs_dict_waves[gauge]['xFRF']:  # then there's data to compare to
                for var in plotVarsWaves:
                    obs_loc = round(obs_dict_waves[gauge]['xFRF'])
                    modVal = hydro[var][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    matchedPlotTime, matchedPlot_obs, matchedPlot_mod = timeMatch(obs_dict_waves[gauge]['time'], obs_dict_waves[gauge][var], times[1:], modVal)
                    if len(matchedPlotTime) > 1:
                        if var == 'Hs':
                            var_name = '$H_{s}$'
                            units = 'm'
                        elif var == 'Tp':
                            var_name = '$T_p$'
                            units = 's'
                        # else;
                        #     raise NotImplementedError('not im')

                            p_dict = {'time': matchedPlotTime,
                                      'obs': matchedPlot_obs,
                                      'model': matchedPlot_mod,
                                      'var_name': var_name,
                                      'units': units,
                                      'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, gauge)}

                            _ = oP.obs_V_mod_TS(figurePath + '{}_{}.png'.format(gauge, var), p_dict)

        for gauge in go.currentsGaugeList:
            print('Gathering Data for currents comparisons on {}'.format(gauge))
            obs_dict_currents[gauge] = go.getCurrents(gauge)
            if 'time' in obs_dict_waves[gauge].keys() and x_n.max() >= obs_dict_waves[gauge]['xFRF']:  # then there's data to compare to
                for var in plotVarsCurrents:
                    if var in ['U']:
                        modVar = 'umean'  # these values come out of the file load and could be streamlined there in the future
                        obsVar = 'aveU'
                    elif var in ['V']:
                        modVar = 'vmean'
                        obsVar = 'aveV'
                    obs_loc = round(obs_dict_currents[gauge]['xFRF'])
                    modVal = hydro[modVar][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                    matchedPlotTime, matchedPlot_obs, matchedPlot_mod = timeMatch(obs_dict_currents[gauge]['time'],
                                                                      obs_dict_currents[gauge][obsVar], times[1:], modVal)
                    if len(matchedPlotTime) > 1:
                        if var == 'U':
                            var_name = '$U$'
                            units = 'm/s'
                        elif var == 'V':
                            var_name = '$V$'
                            units = 'm/s'
                        # else;
                        #     raise NotImplementedError('not im')

                        p_dict = {'time': matchedPlotTime,
                                  'obs': matchedPlot_obs,
                                  'model': matchedPlot_mod,
                                  'var_name': var_name,
                                  'units': units,
                                  'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, gauge)}

                        _ = oP.obs_V_mod_TS(figurePath + '{}_{}.png'.format(gauge, var), p_dict)


        #######################################
        # plot general output from the model
        #######################################
        # QA/QC plots...!!!!
        # 1 - BC Plot
        p_dict = {'time': times[1:], 'x': morpho['x'][0], 'zb': morpho['zb'][0],
                  'init_bathy_stime': meta['bathy_surv_stime'],   'WL': bc['strm_tide_offshore'],
                  'Hs': bc['Hs_offshore'],  'angle': bc['angle_offshore'], 'Tp': bc['Tp_offshore'],
                  'p_title': '%s CSHORE %s - Boundary Conditions' % (version_prefix, startTime)}
        # TODO: this plot can probably take in the observations to show that model input is equal to model output at boundary
        oP.bc_plot(figurePath +'bc.png' , p_dict)

        # 2 - obs V mod bathy
        var_name = 'Bathymetry'
        p_dict = {'x': x_n,  'obs': morpho['zb'][0], 'obs_time': times[0], 'model': morpho['zb'][-1],
                  'model_time': model_time,  'Hs': hydro['Hs'][-1],  'sigma_Hs': np.nanstd(hydro['Hs'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'WL': bc['strm_tide_offshore'],     # do I want to include this?!? it is specific to bathymetry plotting!!!
                  'time': times[1:],
                  'var_name': var_name,
                  'units': 'm',
                  'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, var_name)}

        oP.obs_V_mod_bathy(figurePath+'obsVmodBathy.png', p_dict, obs_dict)

        # 3 - general model results

        p_dict = {'x': x_n,
                  'zb_m': morpho['zb'][-1],
                  'sigma_zbm': np.nanstd(morpho['zb'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'model_time': model_time,
                  'Hs_m': hydro['Hs'][-1],
                  'sigma_Hs': np.nanstd(hydro['Hs'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'setup_m': hydro['mwl'][-1] - hydro['swl'][-1],
                  'sigma_setup': np.nanstd(hydro['mwl'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'p_title': '%s CSHORE %s - Model Results' % (version_prefix, startTime),
                  'time': times[1:]}

        oP.mod_results(figurePath+'modResults.png', p_dict, obs_dict)

        # 4 - alongshore current plot
        p_dict = {'x': x_n,
                  'zb_m': morpho['zb'][-1],
                  'model_time': model_time,
                  'vmean_m': hydro['vmean'][-1],
                  'sigma_vm': hydro['vstd'][-1],
                  'Hs_m': hydro['Hs'][-1],
                  'sigma_Hs': np.nanstd(hydro['Hs'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'p_title': '%s CSHORE %s - Alongshore Current' % (version_prefix, startTime),
                  'time': times[1:]}
        # TODO would be nice if this handled other altimeter data
        oP.als_results(figurePath+'als.png', p_dict, obs_dict)

        # 5 sig wave height at all gages!!!!
        # 6 alongshore current at all gages!!!!

        if lidar['TS_toggle']:
            comp_time_n, obs_runupMean_n, mod_runupMean_n = timeMatch(lidar['runupTime'], lidar['runupMean'], times[1:], hydro['runup_mean'])
            if len(comp_time_n) > 1:
                p_dict = {'time': comp_time_n,
                          'obs': obs_runupMean_n,
                          'model': mod_runupMean_n,
                          'var_name': 'Mean Run-up',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'LiDAR')}

                oP.obs_V_mod_TS(figurePath + 'runupMean.png', p_dict)

            # 2% runup
            comp_time_n, obs_runup_n, mod_runup_n = timeMatch(lidar['runupTime'], lidar['runup2perc'], times[1:], hydro['runup_2_percent'])
            if len(comp_time_n) > 1:
                p_dict = {'time': comp_time_n,
                          'obs': obs_runup_n,
                          'model': mod_runup_n,
                          'var_name': '$2\%$ Exceedance Run-up',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'LiDAR')}

                oP.obs_V_mod_TS(figurePath + 'runup2perc.png', p_dict)

