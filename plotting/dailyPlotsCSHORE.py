import sys, os, glob, argparse, matplotlib
matplotlib.use('Agg')  # for cron
sys.path.append('../')
import numpy as np
import netCDF4 as nc
import datetime as DT
import plotting.operationalPlots as oP
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from getdatatestbed import getDataFRF
from testbedutils.sblib import timeMatch_altimeter, makegif, timeMatch
from prepdata import prepDataLib

# PARAMETERS
workDir = '/home/number/cmtb/liveDataPlots'
endTime = DT.datetime.strptime('2016-05-02T00:15:00Z', '%Y-%m-%dT%H:%M:%SZ')
prefixList = ['MOBILE_RESET']
altStations = ['Alt05', 'Alt04', 'Alt03']
curStations = ['adop-3.5m', 'awac-4.5', 'awac-6m', 'awac-8m']
waveOnlyStations = ['xp100m', 'xp125m', 'xp150m', 'xp200m']
logo_path = '../ArchiveFolder/CHL_logo.png'

# MAIN CODE
def main():

    endTime, prefixList, workDir = getUsrInp()
    if not os.path.exists(workDir):
        os.makedirs(workDir)
    for prefix in prefixList:
        dataLoc = 'morphModels/CSHORE/{0}/{0}.ncml'.format(prefix)
        ncFile, allEpoch = getDataFRF.getnc(dataLoc, 'FRF', 'getDataTestBed')
        ind = np.argwhere(np.diff(ncFile['bathymetryDate']) > 0).squeeze() + 1
        bathyTimes = nc.num2date(ncFile['bathymetryDate'][ind], 
                            ncFile['bathymetryDate'].units)
        ini1Ind = ind[bathyTimes <= endTime][-2]
        ini2Ind = ind[bathyTimes <= endTime][-1]
        if np.any(bathyTimes > endTime):
            finInd = ind[bathyTimes > endTime][0]
            finBathyTime = nc.num2date(ncFile['bathymetryDate'][finInd], ncFile['bathymetryDate'].units)
            finBathy = ncFile['bottomElevation'][finInd].data
        else:
            finBathyTime = []
            finBathy = []

        ini1BathyTime = nc.num2date(ncFile['bathymetryDate'][ini1Ind], ncFile['bathymetryDate'].units)
        ini2BathyTime = nc.num2date(ncFile['bathymetryDate'][ini2Ind], ncFile['bathymetryDate'].units)

        ini1Bathy = ncFile['bottomElevation'][ini1Ind].data
        ini2Bathy = ncFile['bottomElevation'][ini2Ind].data

        start1Time = nc.num2date(ncFile['time'][ini1Ind], ncFile['bathymetryDate'].units)
        start2Time = nc.num2date(ncFile['time'][ini2Ind], ncFile['bathymetryDate'].units)

        print('Survey times:')
        print(ini1BathyTime)
        print(ini2BathyTime)
        print(finBathyTime)
        print('Model times:')
        print(start1Time)
        print(start2Time)
        print(endTime)

        makeGifs(start1Time, start2Time, ini1BathyTime, ini2BathyTime, ini1Bathy,
                 ini2Bathy, prefix, workDir)
        makeGifs(start2Time, endTime, ini2BathyTime, finBathyTime, ini2Bathy, 
                 finBathy, prefix, workDir)
        makeTS(start2Time, endTime, prefix, workDir)

# SUBROUTINES
def getUsrInp():
    parser = argparse.ArgumentParser()
    parser.add_argument('endTime', type=str,
                        help='ending datetime in the format like 2018-04-01T15:00:00Z')
    parser.add_argument('-prefix', type=str,
                        help=('list of model prefixes to process from, e.g. "[MOBILE_RESET]"'))
    parser.add_argument('-workDir', type=str,
                        help='path where the plots will be saved')
    args = parser.parse_args()
    args.endTime = DT.datetime.strptime(args.endTime, '%Y-%m-%dT%H:%M:%SZ')
    if args.prefix == None:
        args.prefix = prefixList
    else:
        args.prefix = eval(args.prefix)
    if args.workDir == None:
        args.workDir = workDir
    return args.endTime, args.prefix, args.workDir

def makeGifs(startTime, endTime, iniBathyTime, finBathyTime, iniBathy, finBathy,
             prefix, workDir):
    gm = getDataFRF.getDataTestBed(startTime, endTime)
    mod = gm.getCSHOREOutput(prefix)
    
    Hs = mod['Hs']
    sigma_Hs = np.nanstd(Hs, 0)
    ylim_Hs = (np.min(Hs - sigma_Hs), np.max(Hs + sigma_Hs))
    setup = mod['setup']
    sigma_setup = np.nanstd(setup, 0)
    ylim_setup = (np.min(setup - sigma_setup), np.max(setup + sigma_setup))
    vmean = mod['aveN']
    sigma_vm = mod['stdN']
    ymin = np.min(vmean - sigma_vm) 
    ymax = np.max(vmean + sigma_vm)
    ylim_V = (ymin - 0.1 * np.abs(ymin), ymax + 0.1 * np.abs(ymax))

    curTime = startTime + DT.timedelta(1)
    while curTime <= endTime:
        gm = getDataFRF.getDataTestBed(curTime - DT.timedelta(1), curTime)
        mod = gm.getCSHOREOutput(prefix)
        if len(mod) == 0:
            curTime += DT.timedelta(1)
            continue
        
        times = mod['time']
        model_time = times[-1]

        altData = [oP.alt_PlotData(station, model_time, times) for 
                       station in altStations]
        curData = [oP.wave_PlotData(station, model_time, times) 
                       for station in curStations]

        for station in altData + curData:
            CSHORETimeMatch(station, mod)

        obs_dict = {'Alt05': altData[altStations.index('Alt05')],
                    'Alt04': altData[altStations.index('Alt04')],
                    'Alt03': altData[altStations.index('Alt03')],
                    'Adopp_35': curData[curStations.index('adop-3.5m')],
                    'AWAC6m': curData[curStations.index('awac-6m')],
                    'AWAC8m': curData[curStations.index('awac-8m')]}

        var_name = 'Bathymetry'
        p_dict = {'x': mod['xFRF'],
                'obs': iniBathy,
                'obs_time': iniBathyTime,
                'obs2': finBathy,
                'obs2_time': finBathyTime,
                'model': mod['zb'][-1],
                'model_time': model_time,
                'Hs': mod['Hs'][-1].filled(np.nan),
                'sigma_Hs': np.nanstd(mod['Hs'], 0),
                'WL': mod['WL'][-1],
                'time': times,
                'var_name': var_name,
                'units': 'm',
                'p_title': '%s CSHORE %s - %s' % (prefix, curTime, var_name)}

        datestring = curTime.strftime('%Y-%m-%dT%H%M%SZ')
        print('Plotting ' + datestring + ' obs_V_mod_bathy')
        oP.obs_V_mod_bathy(os.path.join(workDir, datestring + '_obsVmodBathy.png'), 
                        p_dict, obs_dict, logo_path)
        
        p_dict['zb_m'] = p_dict['model']
        p_dict['sigma_zbm'] = np.nanstd(mod['zb'], 0)
        p_dict['Hs_m'] = p_dict['Hs']
        p_dict['setup_m'] = mod['setup'][-1]
        p_dict['sigma_setup'] = np.nanstd(mod['setup'], 0)
        p_dict['p_title'] = '%s CSHORE %s - Model Results' % (prefix, curTime)

        print('Plotting ' + datestring + ' mod_results')
        oP.mod_results(os.path.join(workDir, datestring + '_modResults.png'), p_dict,
                       obs_dict, ylims=[ylim_Hs, ylim_setup])
        
        p_dict['vmean_m'] = mod['aveN'][-1]
        p_dict['sigma_vm'] = mod['stdN'][-1]
        p_dict['p_title'] = '%s CSHORE %s - Alongshore Current' % (prefix, startTime)

        print('Plotting ' + datestring + ' als')
        oP.als_results(os.path.join(workDir, datestring + '_als.png'), p_dict, obs_dict, 
                                    ylims=[ylim_V, ylim_Hs])

        curTime += DT.timedelta(1)

    datestring = (startTime.strftime('%Y-%m-%dT%H%M%SZ') + '_' +
                endTime.strftime('%Y-%m-%dT%H%M%SZ'))
    imList = sorted(glob.glob(workDir + '/*_obsVmodBathy.png'))
    ofname = os.path.join(workDir, 'obsVmodBathy_{}.gif'.format(datestring))
    makegif(imList, ofname, dt=1.0)
    [os.remove(im) for im in imList]

    imList = sorted(glob.glob(workDir + '/*_modResults.png'))
    ofname = os.path.join(workDir, 'modResults_{}.gif'.format(datestring))
    makegif(imList, ofname, dt=1.0)
    [os.remove(im) for im in imList]

    imList = sorted(glob.glob(workDir + '/*_als.png'))
    ofname = os.path.join(workDir, 'als_{}.gif'.format(datestring))
    makegif(imList, ofname, dt=1.0)
    [os.remove(im) for im in imList]

def makeTS(startTime, endTime, prefix, workDir):
    gm = getDataFRF.getDataTestBed(startTime, endTime)
    mod = gm.getCSHOREOutput(prefix)
    times = mod['time']
    model_time = times[-1]
    altStations_ = [oP.alt_PlotData(station, model_time, times) for 
                   station in altStations]
    curStations_ = [oP.wave_PlotData(station, model_time, times) 
                   for station in curStations]
    waveOnlyStations_ = [oP.wave_PlotData(station, model_time, times) 
                   for station in waveOnlyStations]
    lidar = oP.lidar_PlotData(times)

    datestring = (startTime.strftime('%Y-%m-%dT%H%M%SZ') + '_' +
                  endTime.strftime('%Y-%m-%dT%H%M%SZ'))

    for station in curStations_ + waveOnlyStations_:
        if not station['TS_toggle']:
            continue
        for varName in ['Hs', 'V']:
            if varName == 'Hs':
                varTime = 'wave_time'
                plotName = '$H_{s}$'
                units = 'm'
                varNameM = 'Hs'
            elif varName == 'V' and station['name'] in curStations:
                varTime = 'cur_time'
                plotName = '$V$'
                units = 'm/s'
                varNameM = 'aveN'
            elif varName == 'V' and station['name'] not in curStations:
                continue
            plotTitle = '%s CSHORE %s - %s' % (prefix, startTime, station['name'])
            mod_Hs = mod[varNameM][:, np.where(abs(mod['xFRF'] - station['xFRF']) == 
                            min(abs(mod['xFRF'] - station['xFRF'])), 1, 0) == 1].squeeze()
            comp_time_n, obs_Hs_n, mod_Hs_n = timeMatch(station[varTime], 
                                                        station[varName], times, mod_Hs)
            p_dict = {'time': comp_time_n,
                      'obs': obs_Hs_n,
                      'model': mod_Hs_n,
                      'var_name': plotName,
                      'units': units,
                      'p_title': plotTitle}
            
            print('Plotting {} {} {} timeseries'.format(datestring, station['name'], varName))
            oP.obs_V_mod_TS(os.path.join(workDir, '{}_{}_{}.png'.format(datestring, 
                            station['name'], varName)), p_dict, logo_path)
    
    for station in altStations_:
        if not station['TS_toggle']:
            continue
        obs_zb = station['zb']
        obs_time = station['time']
        obs_loc = round(station['xFRF'])
        mod_zb = mod['zb'][:, np.where(abs(mod['xFRF'] - station['xFRF']) == 
                        min(abs(mod['xFRF'] - station['xFRF'])), 1, 0) == 1].squeeze()
        comp_time = times
        prepData = prepDataLib.PrepDataTools()
        matchObs = prepData.prep_obs2mod(obs_time, obs_zb, comp_time)

        if np.sum(matchObs['mask']) > 0:
            mod_zb = mod_zb[np.where(~matchObs['mask'])]

        p_dict = {'time': matchObs['time'],
                  'obs': matchObs['meanObs'],
                  'model': mod_zb,
                  'var_name': 'Bottom Elevation',
                  'units': 'm',
                  'p_title': '%s CSHORE %s - %s' % (prefix, startTime, 'Alt-04')}

        if np.size(p_dict['obs']) >= 2:
            print('Plotting {} {} {} timeseries'.format(datestring, station['name'], 'BE'))
            oP.obs_V_mod_TS(os.path.join(workDir, '{}_{}_{}.png'.format(datestring, 
                            station['name'], '_BE')), p_dict, logo_path)

    if not  lidar['TS_toggle']:
        return
    for varName in ['runupMean', 'runup2perc']:
        obs_runup = lidar[varName]
        obs_time = lidar['runupTime']
        mod_runup = mod[varName]
        comp_time = times
        comp_time_n, obs_runup_n, mod_runup_n = timeMatch(obs_time, obs_runup, 
                                                          comp_time, mod_runup)
        if len(comp_time_n) <= 1:
            pass
        else:
            if varName == 'runupMean':
                plotName = 'Mean Run-up'
            elif varName == 'runup2perc':
                plotName = '$2\%$ Exceedance Run-up'
            p_dict = {'time': comp_time_n,
                      'obs': obs_runup_n,
                      'model': mod_runup_n,
                      'var_name': plotName,
                      'units': 'm',
                      'p_title': '%s CSHORE %s - %s' % (prefix, startTime, 'LiDAR')}
            print('Plotting {} lidar {} timeseries'.format(datestring, varName))
            oP.obs_V_mod_TS(os.path.join(workDir, '{}_lidar_{}.png'.format(datestring, 
                            varName)), p_dict, logo_path)

def CSHORETimeMatch(obs, mod):
    if obs['TS_toggle'] is False:
        return
    if 'zb' in obs.keys():
        varData = 'zb'
        varTime = 'time'
    elif 'Hs' in obs.keys():
        varData = 'Hs'
        varTime = 'wave_time'
    obs_data = obs[varData]
    obs_time = obs[varTime]
    obs_loc = round(obs['xFRF'])
    mod_data = mod[varData][:, np.where(abs(mod['xFRF'] - obs_loc) == min(
               abs(mod['xFRF'] - obs_loc)), 1, 0) == 1].squeeze()
    comp_time = mod['time']
    comp_time_n, obs_n, mod_n = timeMatch_altimeter(obs_time, 
                                obs_data, comp_time, mod_data)
    plot_ind = np.where(abs(comp_time_n - mod['time'][-1]) == min(
               abs(comp_time_n - mod['time'][-1])), 1, 0)
    obs[varData] = obs_n
    obs[varTime] = comp_time_n
    obs['plot_ind'] = plot_ind

if __name__ == '__main__':
    main()