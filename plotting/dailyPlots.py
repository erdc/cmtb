# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:11:00 2018
This makes observations vs model station plots for a user-specified number of days

@author: Chuan Li
@contact: liC@oregonstate.edu
@organization: USACE CHL FRF

"""
import sys, getopt, warnings, os, glob, argparse, shutil
import numpy as np
import datetime as DT
import netCDF4 as nc
sys.path.append('../')
from getdatatestbed import getDataFRF
from testbedutils import waveLib as sbwave
from testbedutils import sblib as sb
from plotting import operationalPlots as oP

# PARAMETERS
modelList = ['STWAVE', 'CMS']
prefixList = {'STWAVE': ['HP', 'FP', 'CB'],
              'CMS': ['HP']}
stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 
               'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m']
fieldVarList = ['waveHs', 'xRadGrad', 'yRadGrad', 'dissipation']
numDays = 7
workDir = '/home/chuan/Documents/Figures/cmtb'
angadj = 70
logo_path = '../ArchiveFolder/CHL_logo.png'

# MAIN CODE
def main():
    startTime, endTime, modelList, prefixList, workDir = getUsrInp()
    datestring = (startTime.strftime('%Y-%m-%dT%H%M%SZ') + '_' +
                  endTime.strftime('%Y-%m-%dT%H%M%SZ'))
    for model in modelList:
        prefixes = prefixList[model]
        for prefix in prefixes:
            fpath = os.path.join(workDir, model, prefix)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            else:
                files = glob.glob(os.path.join(fpath, '*'))
                for f in files:
                    if os.path.isfile(f): os.remove(f)
                    if os.path.isdir(f): shutil.rmtree(f)
            # Do stations first
            go = getDataFRF.getObs(startTime, endTime)
            gm = getDataFRF.getDataTestBed(startTime, endTime)
            for station in stationList:
                (time, obsStats, modStats, 
                plotList, obsi, modi) = getStats(startTime, endTime, model, prefix, 
                                                station, go, gm)
                if time is None:
                    continue
                for param in plotList:
                    obs = obsStats[param][obsi.astype(np.int)]
                    obs = checkMask(obs, 'observation')
                    mod = modStats[param][modi.astype(np.int)]
                    mod = checkMask(mod, 'model')
                    ofname = os.path.join(fpath, 'CMTB-waveModels_{}_{}_station_{}_{}_{}.png'
                                        .format(model, prefix, station, param, datestring))
                    makePlots(ofname, param, time, obs, mod)
            # Now do field
            gm = getDataFRF.getDataTestBed(startTime, endTime)
            for isLocal in [True, False]:
                try:
                    bathy = gm.getModelField('bathymetry', prefix, isLocal, model=model)
                except AssertionError as err:
                    if err.message == " there's no data":
                        local = ' local' if isLocal is True else ' regional'
                        print('No ' + model + local + ' ' + prefix + ' field data for ' + datestring)
                        continue
                    else:
                        raise err
                for varName in fieldVarList:
                    try:
                        var = gm.getModelField(varName, prefix, isLocal, model=model)
                    except AssertionError as err:
                        if 'variable called is not in file please use' in err.message:
                            continue
                        else:
                            raise err
                    for key in var.keys(): # Check data for masks
                        var[key] = checkMask(var[key], key + ' (in ' + varName + ')')
                    fieldpacket = makeFieldpacket(varName, var, isLocal)
                    grid = 'Local' if isLocal else 'Regional'
                    kwargs = {}
                    if varName == 'waveHs':
                        waveDm = gm.getModelField('waveDm', prefix, isLocal, model=model)['waveDm']
                        kwargs['directions'] = waveDm
                    print('Generating field variable plots for ' + prefix + ' ' + 
                        model + ' ' + grid + ' ' + varName)
                    oP.plotSpatialFieldData(bathy, fieldpacket, os.path.join(fpath, varName), 
                                                isLocal, model=model, **kwargs)
                    imList = sorted(glob.glob(fpath + '/' + varName + '_*.png'))
                    print('Generating field variable gifs for ' + prefix + ' ' + model 
                        + ' ' + grid + ' ' + varName)
                    ofname = os.path.join(fpath, 'CMTB-waveModels_{}_{}_{}_{}_{}.gif'
                                        .format(model, prefix, grid, varName, datestring))
                    sb.makegif(imList, ofname)
                    [os.remove(ff) for ff in imList]

# SUBROUTINES
def makeFieldpacket(varName, var, isLocal):
    grid = 'Local FRF North Property' if isLocal else 'Regional Grid'
    fieldpacket = {'xlabel': 'Longshore distance [m]', 
                   'ylabel': 'Cross-shore distance [m]',
                   'field': var[varName], 'xcoord': var['xFRF'], 
                   'ycoord': var['yFRF'], 'time': var['time']}
    if varName == 'waveHs':
        fieldpacket['title'] = grid + ': Significant wave height $H_s$'
        fieldpacket['cblabel'] = 'Wave height $H_s [m]$'
    elif varName == 'xRadGrad':
        fieldpacket['title'] = grid + 'Radiation Stress Gradients - X'
        fieldpacket['cblabel'] = 'Radiation Stress Gradients - X'
    elif varName == 'yRadGrad':
        fieldpacket['title'] = grid + 'Radiation Stress Gradients - Y'
        fieldpacket['cblabel'] = 'Radiation Stress Gradients - Y'
    elif varName == 'dissipation':
        fieldpacket['title'] = grid + 'Wave Dissipation'
        fieldpacket['cblabel'] = 'Wave Dissipation'
    return fieldpacket

def checkMask(var, varname):
    if isinstance(var, np.ma.MaskedArray):
        if np.ma.is_masked(var):
            print('Warning: there are masked values in ' + varname)
        return var.data
    else:
        return var

def getUsrInp():
    parser = argparse.ArgumentParser()
    parser.add_argument('endTime', type=str,
                        help='ending datetime in the format like 2018-04-01T15:00:00Z')
    parser.add_argument('-numDays', type=int,
                        help='number of (integer) days to process prior to endTime')
    parser.add_argument('-model', type=str,
                        help='list of numerical model to process from, e.g. "[STWAVE, CMS]"')
    parser.add_argument('-prefix', type=str,
                        help=('dictionary of prefixes for the each numerical model, e.g.'
                              '"' + "{'STWAVE': ['HP', 'FP', 'CB'], 'CMS': ['HP']}" + '"'))
    parser.add_argument('-workDir', type=str,
                        help='path where the plots will be saved')
    args = parser.parse_args()
    args.endTime = DT.datetime.strptime(args.endTime, '%Y-%m-%dT%H:%M:%SZ')
    if args.numDays == None:
        args.numDays= numDays
    if args.model == None:
        args.model = modelList
    elif args.model[0] == '[' and args.model[-1] == ']':
        args.model = eval(args.model)
    else:
        args.model = [args.model]
    if args.prefix == None:
        args.prefix = prefixList
    else:
        args.prefix = eval(args.prefix)
    if args.workDir == None:
        args.workDir = workDir
    for model in args.model:
        if model not in args.prefix.keys():
            sys.exit('Error: ' + model + ' prefixes not provided')
    startTime = args.endTime - DT.timedelta(days=numDays)
    
    return startTime, args.endTime, args.model, args.prefix, args.workDir

def getStats(startTime, endTime, model, prefix, station, go, gm):
    # Get observation data
    wo = go.getWaveSpec(station)
    if 'time' not in wo:
        return None, None, None, None, None, None
    if station in go.directional: # directionalWaveGaugeList spectra
        if prefix is not 'FP': # half plane
            wo['dWED'], wo['wavedirbin'] = sbwave.HPchop_spec(wo['dWED'], 
            wo['wavedirbin'], angadj=angadj)
        obsStats = sbwave.waveStat(wo['dWED'], wo['wavefreqbin'], wo['wavedirbin'])
    else: # non-directionalWaveGaugeList
        obsStats = sbwave.stats1D(wo['fspec'], wo['wavefreqbin'])
    # Get model results
    with warnings.catch_warnings(record=True) as w:
        wm = gm.getWaveSpecModel(prefix, station, model=model)
        if wm == None:
            return [None] * 6
        if w != []:
            assert str(w[0].message) == (
                "Warning: 'partition' will ignore the 'mask' of the MaskedArray."), w[0].message
    modStats = sbwave.waveStat(wm['dWED'], wm['wavefreqbin'], wm['wavedirbin'])
    # Time match
    time, obsi, modi = sb.timeMatch(wo['epochtime'], np.arange(wo['time'].shape[0]),
                                    wm['epochtime'], np.arange(wm['time'].shape[0]))
    if time.size == 0:
        print('No matching time between observation and model results for station ' + station)
        return [None] * 6
    if station in go.directional:
        plotList = ['Hm0', 'Tm', 'sprdF', 'sprdD', 'Tp', 'Dm']
    else:
        plotList = ['Hm0', 'Tm', 'sprdF', 'Tp']
    return time, obsStats, modStats, plotList, obsi, modi

def makePlots(ofname, param, time, obs, mod):
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
    dataDict = {'time': nc.num2date(time, 'seconds since 1970-01-01'),
                'obs': obs,
                'model': mod,
                'var_name': param,
                'units': units,
                'p_title': title}
    print('Plotting ' + ofname)
    with warnings.catch_warnings(record=True) as w:
        oP.obs_V_mod_TS(ofname, dataDict, logo_path=logo_path)
        if w != []:
            assert str(w[0].message) == (
                'This figure includes Axes that are not compatible ' +
                'with tight_layout, so results might be incorrect.')

if __name__ == '__main__':
    main()