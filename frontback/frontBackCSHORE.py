from prepdata import inputOutput, prepDataLib
import os
import datetime as DT
import netCDF4 as nc
import numpy as np
from getdatatestbed.getDataFRF import getObs, getDataTestBed
from testbedutils.sblib import timeMatch, timeMatch_altimeter, makeNCdir
from testbedutils.anglesLib import STWangle2geo, vectorRotation
import plotting.operationalPlots as oP
import makenc
from subprocess import check_output
import prepdata.prepDataLib as PDL

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

    # get into the directory I need
    start_dir = workingDir
    path_prefix = os.path.join(model, version_prefix)  # data super directiory
    d_s = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    date_str = d_s.strftime('%Y-%m-%dT%H%M%SZ') # THE COLONS!!! startTime has COLONS!!!

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
    BC_FRFX = meta["BC_FRF_X"]
    BC_FRFY = meta["BC_FRF_Y"]
    x_n = BC_FRFX - morpho['x'][0]
    model_time = times[-1]

    # convert model time to epoch?
    # timeunits = 'seconds since 1970-01-01 00:00:00'
    # model_timeN = nc.date2num(model_time, timeunits)

    # make the plots like a boss, with greatness
    if pFlag:

        if 'MOBILE' in version_prefix:

            # A - pull all the the observations that I need and store as dictionaries!!!!!!!
            # Altimeter data!!!!!!!!
            Alt05 = oP.alt_PlotData('Alt05', model_time, times)
            Alt04 = oP.alt_PlotData('Alt04', model_time, times)
            Alt03 = oP.alt_PlotData('Alt03', model_time, times)

            # go ahead and time match the altimeter data
            if Alt05['TS_toggle']:
                # ALT05
                obs_zb = Alt05['zb']
                obs_time = Alt05['time']
                obs_loc = round(Alt05['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]
                comp_time_n, obs_n, mod_n = timeMatch_altimeter(obs_time, obs_zb, comp_time, mod_zb)
                plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
                # delete and re-assign
                del Alt05['zb']
                del Alt05['time']
                del Alt05['plot_ind']
                Alt05['zb'] = obs_n
                Alt05['time'] = comp_time_n
                Alt05['plot_ind'] = plot_ind

            if Alt04['TS_toggle']:
                # ALT04
                obs_zb = Alt04['zb']
                obs_time = Alt04['time']
                obs_loc = round(Alt04['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]
                comp_time_n, obs_n, mod_n = timeMatch_altimeter(obs_time, obs_zb, comp_time, mod_zb)
                plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
                # delete and re-assign
                del Alt04['zb']
                del Alt04['time']
                del Alt04['plot_ind']
                Alt04['zb'] = obs_n
                Alt04['time'] = comp_time_n
                Alt04['plot_ind'] = plot_ind

            if Alt03['TS_toggle']:
                # ALT03
                obs_zb = Alt03['zb']
                obs_time = Alt03['time']
                obs_loc = round(Alt03['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]
                comp_time_n, obs_n, mod_n = timeMatch_altimeter(obs_time, obs_zb, comp_time, mod_zb)
                plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
                # delete and re-assign
                del Alt03['zb']
                del Alt03['time']
                del Alt03['plot_ind']
                Alt03['zb'] = obs_n
                Alt03['time'] = comp_time_n
                Alt03['plot_ind'] = plot_ind




        # wave data & current data!!!
        Adopp_35 = oP.wave_PlotData('adop-3.5m', model_time, times)
        AWAC6m = oP.wave_PlotData('awac-6m', model_time, times)
        # this is just to check to see if I rounded down when i set my bathymetry,
        # in which case the 6m AWAC would not be inside the plot limits.
        if AWAC6m['xFRF'] > max(x_n):
            # if it is, round it down to nearest 1m - this will move it to the boundary if it IS the boundary gage
            AWAC6m['xFRF'] = float(int(AWAC6m['xFRF']))
        AWAC8m = oP.wave_PlotData('awac-8m', model_time, times)
        # this is just to check to see if I rounded down when i set my bathymetry,
        # in which case the 8m AWAC would not be inside the plot limits.
        if AWAC8m['xFRF'] > max(x_n):
            # if it is, round it down to nearest 1m - this will move it to the boundary if it IS the boundary gage
            AWAC8m['xFRF'] = float(int(AWAC8m['xFRF']))

        # go ahead and time match the wave and current data!

        if Adopp_35['TS_toggle']:
            # Adopp_35
            # get time-matched data!!!!!! waves
            obs_Hs = Adopp_35['Hs']
            obs_time = Adopp_35['wave_time']
            obs_loc = round(Adopp_35['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)
            plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del Adopp_35['Hs']
            del Adopp_35['wave_time']
            del Adopp_35['plot_ind']
            Adopp_35['Hs'] = obs_Hs_n
            Adopp_35['wave_time'] = comp_time_n
            Adopp_35['plot_ind'] = plot_ind

            # get time-matched data!!!!!! currents
            # V
            obs_V = Adopp_35['V']
            obs_time = Adopp_35['cur_time']
            obs_loc = round(Adopp_35['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)
            plot_ind_V = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del Adopp_35['V']
            temp_cur_time = Adopp_35['cur_time']
            del Adopp_35['cur_time']
            del Adopp_35['plot_ind_V']
            Adopp_35['V'] = obs_V_n
            Adopp_35['cur_time'] = comp_time_n
            Adopp_35['plot_ind_V'] = plot_ind_V

            # U
            obs_U = Adopp_35['U']
            obs_time = temp_cur_time
            obs_loc = round(Adopp_35['xFRF'])
            mod_U = hydro['umean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_U_n, mod_U_n = timeMatch(obs_time, obs_U, comp_time, mod_U)
            # delete and re-assign
            del Adopp_35['U']
            Adopp_35['U'] = obs_U_n

        if AWAC6m['TS_toggle']:
            # AWAC6m
            # get time-matched data!!!!!! waves
            obs_Hs = AWAC6m['Hs']
            obs_time = AWAC6m['wave_time']
            obs_loc = round(AWAC6m['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)
            plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del AWAC6m['Hs']
            del AWAC6m['wave_time']
            del AWAC6m['plot_ind']
            AWAC6m['Hs'] = obs_Hs_n
            AWAC6m['wave_time'] = comp_time_n
            AWAC6m['plot_ind'] = plot_ind

            # get time-matched data!!!!!! currents
            # V
            obs_V = AWAC6m['V']
            obs_time = AWAC6m['cur_time']
            obs_loc = round(AWAC6m['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)
            plot_ind_V = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del AWAC6m['V']
            temp_cur_time = AWAC6m['cur_time']
            del AWAC6m['cur_time']
            del AWAC6m['plot_ind_V']
            AWAC6m['V'] = obs_V_n
            AWAC6m['cur_time'] = comp_time_n
            AWAC6m['plot_ind_V'] = plot_ind_V

            # U
            obs_U = AWAC6m['U']
            obs_time = temp_cur_time
            obs_loc = round(AWAC6m['xFRF'])
            mod_U = hydro['umean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_U_n, mod_U_n = timeMatch(obs_time, obs_U, comp_time, mod_U)
            # delete and re-assign
            del AWAC6m['U']
            AWAC6m['U'] = obs_U_n

        if AWAC8m['TS_toggle']:
            # AWAC8m
            # get time-matched data!!!!!! waves
            obs_Hs = AWAC8m['Hs']
            obs_time = AWAC8m['wave_time']
            obs_loc = round(AWAC8m['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)
            plot_ind = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del AWAC8m['Hs']
            del AWAC8m['wave_time']
            del AWAC8m['plot_ind']
            AWAC8m['Hs'] = obs_Hs_n
            AWAC8m['wave_time'] = comp_time_n
            AWAC8m['plot_ind'] = plot_ind

            # get time-matched data!!!!!! currents
            # V
            obs_V = AWAC8m['V']
            obs_time = AWAC8m['cur_time']
            obs_loc = round(AWAC8m['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)
            plot_ind_V = np.where(abs(comp_time_n - model_time) == min(abs(comp_time_n - model_time)), 1, 0)
            # delete and re-assign
            del AWAC8m['V']
            temp_cur_time = AWAC8m['cur_time']
            del AWAC8m['cur_time']
            del AWAC8m['plot_ind_V']
            AWAC8m['V'] = obs_V_n
            AWAC8m['cur_time'] = comp_time_n
            AWAC8m['plot_ind_V'] = plot_ind_V

            # U
            obs_U = AWAC8m['U']
            obs_time = temp_cur_time
            obs_loc = round(AWAC8m['xFRF'])
            mod_U = hydro['umean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_U_n, mod_U_n = timeMatch(obs_time, obs_U, comp_time, mod_U)
            # delete and re-assign
            del AWAC8m['U']
            AWAC8m['U'] = obs_U_n



        # LiDAR stuff goes here...
        lidar = oP.lidar_PlotData(times)

        if 'MOBILE' in version_prefix:
            obs_dict = {'Alt05': Alt05,
                        'Alt04': Alt04,
                        'Alt03': Alt03,
                        'Adopp_35': Adopp_35,
                        'AWAC6m': AWAC6m,
                        'AWAC8m': AWAC8m,
                        'lidar': lidar,
                        }
        else:
            obs_dict = {'Adopp_35': Adopp_35,
                        'AWAC6m': AWAC6m,
                        'AWAC8m': AWAC8m,
                        'lidar': lidar,
                        }


        # QA/QC plots...!!!!
        # 1 - BC Plot
        path = os.path.join(start_dir, path_prefix, date_str, 'figures/')
        p_dict = {'time': times[1:],
                  'x': morpho['x'][0],
                  'zb': morpho['zb'][0],
                  'init_bathy_stime': meta['bathy_surv_stime'],
                  'WL': bc['strm_tide_offshore'],
                  'Hs': bc['Hs_offshore'],
                  'angle': bc['angle_offshore'],
                  'Tp': bc['Tp_offshore'],
                  'p_title': '%s CSHORE %s - Boundary Conditions' % (version_prefix, startTime)}

        oP.bc_plot(path +'bc.png' , p_dict)

        # 2 - obs V mod bathy
        var_name = 'Bathymetry'
        p_dict = {'x': x_n,
                  'obs': morpho['zb'][0],
                  'obs_time': times[0],
                  'model': morpho['zb'][-1],
                  'model_time': model_time,
                  'Hs': hydro['Hs'][-1],
                  'sigma_Hs': np.nanstd(hydro['Hs'], 0),
                  # dont worry about the runtime warning, it just means you have all nans in that x position, so it will return nans!
                  'WL': bc['strm_tide_offshore'],
                  # do I want to include this?!? it is specific to bathymetry plotting!!!
                  'time': times[1:],
                  'var_name': var_name,
                  'units': 'm',
                  'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, var_name)}

        oP.obs_V_mod_bathy(path+'obsVmodBathy.png', p_dict, obs_dict)

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

        oP.mod_results(path+'modResults.png', p_dict, obs_dict)

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

        oP.als_results(path+'als.png', p_dict, obs_dict)

        # now we want time-series comparisons at all my gages!!!!!!!
        # this is necessary for the preservation of the empire

        # 5 sig wave height at all gages!!!!
        # 6 alongshore current at all gages!!!!

        # adopp_35
        if Adopp_35['TS_toggle']:

            # 5 a) adopp_35 Hs
            # get time-matched data!!!!!!
            obs_Hs = Adopp_35['Hs']
            obs_time = Adopp_35['wave_time']
            obs_loc = round(Adopp_35['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_Hs_n,
                          'model': mod_Hs_n,
                          'var_name': '$H_{s}$',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'Adopp 3.5')}

                oP.obs_V_mod_TS(path+'adop35_Hs.png', p_dict)

            # 6 a) adopp_35 V
            # get time-matched data!!!!!!
            obs_V = Adopp_35['V']
            obs_time = Adopp_35['cur_time']
            obs_loc = round(Adopp_35['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_V_n,
                          'model': mod_V_n,
                          'var_name': '$V$',
                          'units': 'm/s',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'Adopp 3.5')}

                oP.obs_V_mod_TS(path+'adop35_V.png', p_dict)


        # AWAC6m
        if AWAC6m['TS_toggle']:

            # 5 b) AWAC6m Hs
            # get time-matched data!!!!!!
            obs_Hs = AWAC6m['Hs']
            obs_time = AWAC6m['wave_time']
            obs_loc = round(AWAC6m['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_Hs_n,
                          'model': mod_Hs_n,
                          'var_name': '$H_{s}$',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'AWAC 6m')}

                oP.obs_V_mod_TS(path + 'AWAC6m_Hs.png', p_dict)

            # 6 b) AWAC6m V
            # get time-matched data!!!!!!
            obs_V = AWAC6m['V']
            obs_time = AWAC6m['cur_time']
            obs_loc = round(AWAC6m['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_V_n,
                          'model': mod_V_n,
                          'var_name': '$V$',
                          'units': 'm/s',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'AWAC 6m')}

                oP.obs_V_mod_TS(path+'AWAC6m_V.png', p_dict)


        # 5 c) AWAC8m Hs
        if AWAC8m['TS_toggle']:

            # 5 c) AWAC8m Hs
            # get time-matched data!!!!!!
            obs_Hs = AWAC8m['Hs']
            obs_time = AWAC8m['wave_time']
            obs_loc = round(AWAC8m['xFRF'])
            mod_Hs = hydro['Hs'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(obs_time, obs_Hs, comp_time, mod_Hs)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_Hs_n,
                          'model': mod_Hs_n,
                          'var_name': '$H_{s}$',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'AWAC 8m')}

                oP.obs_V_mod_TS(path + 'AWAC8m_Hs.png', p_dict)

            # 6 c) AWAC8m V
            #  get time-matched data!!!!!!
            obs_V = AWAC8m['V']
            obs_time = AWAC8m['cur_time']
            obs_loc = round(AWAC8m['xFRF'])
            mod_V = hydro['vmean'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
            comp_time = times[1:]
            comp_time_n, obs_V_n, mod_V_n = timeMatch(obs_time, obs_V, comp_time, mod_V)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_V_n,
                          'model': mod_V_n,
                          'var_name': '$V$',
                          'units': 'm/s',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'AWAC 8m')}

                oP.obs_V_mod_TS(path + 'AWAC8m_V.png', p_dict)

        if 'MOBILE' in version_prefix:
            # 7 bottom elevation at all gages!!!!

            # 7 a) alt 05 bottom elevation
            if Alt05['TS_toggle']:
                # get time-matched data!!!!!!
                obs_zb = Alt05['zb']
                obs_time = Alt05['time']
                obs_loc = round(Alt05['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]

                prepData = prepDataLib.PrepDataTools()
                matchObs = prepData.prep_obs2mod(obs_time, obs_zb, comp_time)

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
                    oP.obs_V_mod_TS(path + 'alt05_BE.png', p_dict)

            # 7 b) alt 04 bottom elevation
            if Alt04['TS_toggle']:
                # get time-matched data!!!!!!
                obs_zb = Alt04['zb']
                obs_time = Alt04['time']
                obs_loc = round(Alt04['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]

                prepData = prepDataLib.PrepDataTools()
                matchObs = prepData.prep_obs2mod(obs_time, obs_zb, comp_time)

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
                    oP.obs_V_mod_TS(path + 'alt04_BE.png', p_dict)

            # 7 c) alt 03 bottom elevation
            if Alt03['TS_toggle']:
                # get time-matched data!!!!!!
                obs_zb = Alt03['zb']
                obs_time = Alt03['time']
                obs_loc = round(Alt03['xFRF'])
                mod_zb = morpho['zb'][:, np.where(abs(x_n - obs_loc) == min(abs(x_n - obs_loc)), 1, 0) == 1].squeeze()
                comp_time = times[1:]

                prepData = prepDataLib.PrepDataTools()
                matchObs = prepData.prep_obs2mod(obs_time, obs_zb, comp_time)

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
                    oP.obs_V_mod_TS(path + 'alt03_BE.png', p_dict)

        # 8 - LiDAR runup comparison plots

        if lidar['TS_toggle']:
            # get time-matched data!!!!!!
            obs_runupMean = lidar['runupMean']
            obs_time = lidar['runupTime']
            mod_runupMean = hydro['runup_mean']
            comp_time = times[1:]
            comp_time_n, obs_runupMean_n, mod_runupMean_n = timeMatch(obs_time, obs_runupMean, comp_time, mod_runupMean)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_runupMean_n,
                          'model': mod_runupMean_n,
                          'var_name': 'Mean Run-up',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'LiDAR')}

                oP.obs_V_mod_TS(path + 'runupMean.png', p_dict)

            # 8 b) 2% runup
            # get time-matched data!!!!!!
            obs_runup = lidar['runup2perc']
            obs_time = lidar['runupTime']
            mod_runup = hydro['runup_2_percent']
            comp_time = times[1:]
            comp_time_n, obs_runup_n, mod_runup_n = timeMatch(obs_time, obs_runup, comp_time, mod_runup)

            if len(comp_time_n) <= 1:
                pass
            else:
                p_dict = {'time': comp_time_n,
                          'obs': obs_runup_n,
                          'model': mod_runup_n,
                          'var_name': '$2\%$ Exceedance Run-up',
                          'units': 'm',
                          'p_title': '%s CSHORE %s - %s' % (version_prefix, startTime, 'LiDAR')}

                oP.obs_V_mod_TS(path + 'runup2perc.png', p_dict)

    # make the nc files
    nc_dict = makeCSHORE_ncdict(startTime=startTime, inputDict=inputDict)
    globalYaml = None

    yaml_dir = os.path.dirname(os.path.abspath(__file__))
    # need to go up one
    yaml_dir = os.path.dirname(yaml_dir)

    if 'MOBILE' in version_prefix:
        globalYaml = os.path.join(yaml_dir, 'yaml_files/CSHORE/CSHORE_mobile_global.yml')
        varYaml = os.path.join(yaml_dir, 'yaml_files/CSHORE/CSHORE_mobile_var.yml')
    elif 'FIXED' in version_prefix:
        globalYaml = os.path.join(yaml_dir, 'yaml_files/CSHORE/CSHORE_fixed_global.yml')
        varYaml = os.path.join(yaml_dir, 'yaml_files/CSHORE/CSHORE_fixed_var.yml')
    else:
        raise NotImplementedError('please check version prefix')

    assert globalYaml is not None, 'CSHORE_analysis Error: Version prefix not recognized'

    NCpath = makeNCdir(netCDFdir, version_prefix, date_str, model='CSHORE')

    # make the name of this nc file your OWN SELF BUM!
    NCname = 'CMTB-morphModels_CSHORE_%s_%s.nc' %(version_prefix, date_str)

    makenc.makenc_CSHORErun(os.path.join(NCpath, NCname), nc_dict, globalYaml, varYaml)

    t = 1

def makeCSHORE_ncdict(startTime,inputDict):
    """

    Args:
        startTime (str): this is the time that all the CSHORE runs are tagged by (e.g., '2012-12-31T00:30:30Z')
        inputDicts (dict): keys are
            version_prefix - right now we have MOBILE, MOBILE_RESET. FIXED
            workingDir - path to the working directory the user wants
    Returns:
         nc_dict (dict): the dictionary with keys that you hand to the nc file
            (the data has been rotated back to the standard FRF conventions)

    """

    version_prefix = inputDict['version_prefix']
    workingDir = inputDict['workingDirectory']
    model = 'CSHORE'
    # initialize the class
    cshore_io = inputOutput.cshoreIO()

    # get into the directory I need
    start_dir = workingDir
    path_prefix = "%s/" % version_prefix  # data super directiory
    d_s = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    date_str = d_s.strftime('%Y-%m-%dT%H%M%SZ')  # THE COLONS!!! startTime has COLONS!!!

    params, bc, veg, hydro, sed, morpho, meta = cshore_io.load_CSHORE_results(os.path.join(start_dir, model, path_prefix, date_str))
    # params - metadata about the run
    # bc - boundary condition data, but for some reason it does not include the initial conditions?
    # veg - vegetation information
    # hydro - current and wave information
    # sed - sediment information
    # morpho - bed elevation information


    # try out the makeCSHORE_ncdict function!!
    nc_dict = {}
    dim_t = len(hydro['umean'])
    dim_x = len(hydro['umean'][0])

    step = hydro['time_end'][1] - hydro['time_end'][0]
    pierang = 71.8
    # get my time stuff!!!
    times = np.array([d_s + DT.timedelta(seconds=s) for s in np.ravel(hydro['time_end'] + step)])
    #  The 1800 will set the time to be IN BETWEEN the start time and end time
    timeunits = 'seconds since 1970-01-01 00:00:00'
    nc_dict['time'] = nc.date2num(times, timeunits)

    # get my cross-shore X (and Y!!!) in FRF coords!!!
    nc_dict['xFRF'] = meta["BC_FRF_X"] - morpho['x'][0]
    nc_dict['yFRF'] = meta["BC_FRF_Y"]

    # get my stuff that needs to be rotated
    test_fun = lambda x: vectorRotation(x, theta=pierang + (90 - pierang) + pierang)
    nc_dict['aveE'] = np.zeros([dim_t, dim_x])
    nc_dict['aveN'] = np.zeros([dim_t, dim_x])
    nc_dict['stdE'] = np.zeros([dim_t, dim_x])
    nc_dict['stdN'] = np.zeros([dim_t, dim_x])
    nc_dict['waveMeanDirection'] = np.zeros([dim_t, dim_x]) + np.nan
    nc_dict['qbx'] = np.zeros([dim_t, dim_x])
    nc_dict['qby'] = np.zeros([dim_t, dim_x])
    nc_dict['qsx'] = np.zeros([dim_t, dim_x])
    nc_dict['qsy'] = np.zeros([dim_t, dim_x])
    for ii in range(0, len(hydro['umean'])):
        # current stuff
        newV = [test_fun(x) for x in zip(hydro['umean'][ii][:], hydro['vmean'][ii][:])]
        nc_dict['aveE'][ii] = zip(*newV)[0]
        nc_dict['aveN'][ii] = zip(*newV)[1]

        newStd = [test_fun(x) for x in zip(hydro['ustd'][ii][:], hydro['vstd'][ii][:])]
        nc_dict['stdE'][ii] = zip(*newStd)[0]
        nc_dict['stdN'][ii] = zip(*newStd)[1]

        # wave angle
        t1 = 360 / float(2 * np.pi) * hydro['stheta'][ii]
        t1[~np.isnan(t1)] = STWangle2geo(t1[~np.isnan(t1)], pierang=pierang)
        nc_dict['waveMeanDirection'][ii] = t1

        # sediment transport rate
        new_qb = [test_fun(x) for x in zip(sed['qbx'][ii][:], sed['qby'][ii][:])]
        nc_dict['qbx'][ii] = zip(*new_qb)[0]
        nc_dict['qby'][ii] = zip(*new_qb)[1]

        new_qs = [test_fun(x) for x in zip(sed['qsx'][ii][:], sed['qsy'][ii][:])]
        nc_dict['qsx'][ii] = zip(*new_qs)[0]
        nc_dict['qsy'][ii] = zip(*new_qs)[1]

    # wave and WL stuff!!!!
    nc_dict['waveHs'] = hydro['Hs']
    nc_dict['waterLevel'] = hydro['mwl']
    nc_dict['stdWaterLevel'] = hydro['sigma']
    nc_dict['setup'] = hydro['mwl'] - hydro['swl']

    # runup stuff!
    nc_dict['runup2perc'] = hydro['runup_2_percent']
    nc_dict['runupMean'] = hydro['runup_mean']

    # other sediment stuff
    nc_dict['probabilitySuspension'] = sed['ps']
    nc_dict['probabilityMovement'] = sed['pb']
    nc_dict['suspendedSedVolume'] = sed['vs']

    # bathymetry
    nc_dict['bottomElevation'] = morpho['zb']  # you may have to screw with this with fixed vs. mobile???
    # if the fixed bed just copies the same bathy to each time-step, you will need to just take the first one!!!
    nc_dict['surveyNumber'] = np.zeros([dim_t]) + meta['bathy_surv_num']
    nc_dict['profileNumber'] = np.zeros([dim_t]) + meta['bathy_prof_num']
    nc_dict['bathymetryDate'] = nc.date2num(np.array([meta['bathy_surv_stime'] + DT.timedelta(hours=0) for i in range(dim_t)]), timeunits)

    # wave flag stuff
    wFlag = meta['blank_wave_data']
    if np.size(wFlag) == 0:
        nc_dict['waveFlag'] = np.zeros(np.shape(nc_dict['time'])).astype(int)
    else:
        nc_dict['waveFlag'] = []
        for tt in range(0, len(times)):
            if times[tt] in wFlag:
                nc_dict['waveFlag'].append(1)
            else:
                nc_dict['waveFlag'].append(0)
        nc_dict['waveFlag'] = np.array(nc_dict['waveFlag']).astype(int)

    return nc_dict

def CSHOREsimSetup(startTime, inputDict):
    """Author: David Young, Master of the Universe
    Association: USACE CHL Field Research Facility
    Project:  Coastal Model Test Bed

    This Function is the master call for the  data preperation for the Coastal Model
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
    time_step = 1 # time step for model in hours
    dx = 1  # cross-shore grid spacing (FRF coord units - m)
    fric_fac = 0.015

    # ______________________________________________________________________________

    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # Time Stuff!
    if type(timerun) == str:
        timerun = int(timerun)
    start_time = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    # scream at them if the simulation does not start on a whole hour!
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
    wave_data8m = frf_DataW.getWaveSpec(gaugenumber=12)
    wave_data6m = frf_DataW.getWaveSpec(gaugenumber=4)
    # getObs instance for bathy!! - only pull something that is on that day!
    frf_DataB = getObs(start_time, end_time + DT.timedelta(days=0, hours=0, minutes=1), server)
    cmtb_data = getDataTestBed(start_time, end_time + DT.timedelta(days=0, hours=0, minutes=1), server)

    # _____________________________________________________________________________
    # ______________Decision Time - Fixed vs. Mobile_______________________________
    if version_prefix == 'FIXED':
        # if it is fixed, first thing to do is get waves

        ## _____________WAVES____________________________
        # which wave gage?
        o_dict = prep.waveTree_CSHORE(wave_data8m, wave_data6m, BC_dict['timebc_wave'], start_time)
        assert o_dict is not None, 'Simulation broken.  Missing wave data!'

        # ____________ BATHY ______________________
        print('\n____________________\nGetting Bathymetric Data\n')

        # what do I use as my initial bathy?
        bathy_loc_List = np.array(['integrated_bathy', 'survey'])
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
            cshore_io_O = inputOutput.cshoreIO()
            # get into the directory I need
            path_prefix_O = path_prefix
            params0, bc0, veg0, hydro0, sed0, morpho0, meta0 = cshore_io_O.load_CSHORE_results(path_prefix_O + Time_O)
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
            # yes i am

            # which wave gage?
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
                cshore_io_O = inputOutput.cshoreIO()
                # get into the directory I need
                path_prefix_O = path_prefix
                params0, bc0, veg0, hydro0, sed0, morpho0, meta0 = cshore_io_O.load_CSHORE_results(path_prefix_O + Time_O)
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
                assert 'zb' in list(b_dict.keys()), 'Please begin simulations the day after a newly collected bathymetry'


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

    # troubleshooting wave direction stuff!
    # BC_dict['Hs'] = 5 * np.ones(np.shape(BC_dict['Hs']))
    # BC_dict['Tp'] = 18 * np.ones(np.shape(BC_dict['Tp']))
    # BC_dict['angle'] = -60 * np.ones(np.shape(BC_dict['angle']))
    # t = 1

    # check to see if I actually have wave data after all this...
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

