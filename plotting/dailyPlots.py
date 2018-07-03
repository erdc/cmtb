# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:11:00 2018
This makes observations vs model station plots for a user-specified number of days

@author: Chuan Li
@contact: liC@oregonstate.edu
@organization: USACE CHL FRF

"""
import sys, getopt, warnings, os, glob
import numpy as np
import datetime as DT
import netCDF4 as nc
sys.path.append('../')
from getdatatestbed import getDataFRF
from testbedutils import waveLib as sbwave
from testbedutils import sblib as sb
from plotting import operationalPlots as oP

# PARAMETERS
stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 
               'awac-6m', 'awac-4.5m', 'adop-3.5m', 'xp200m', 'xp150m', 'xp125m']
fieldVarList = ['waveHs', 'xRadGrad', 'yRadGrad', 'dissipation']
# fieldVarList = ['waveHs']
angadj = 70
logo_path = '../ArchiveFolder/CHL_logo.png'

# MAIN CODE
def main():
    start = DT.datetime.now()

    startTime, endTime, model, prefix, workingDirectory = getUsrInp()
    datestring = (startTime.strftime('%Y-%m-%dT%H%M%SZ') + '_' +
                  endTime.strftime('%Y-%m-%dT%H%M%SZ'))
    fpath = os.path.join(workingDirectory, model, prefix, datestring)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
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
        bathy = gm.getStwaveField('bathymetry', prefix, isLocal, model=model)
        for varName in fieldVarList:
            try:
                var = gm.getStwaveField(varName, prefix, isLocal, model=model)
            except AssertionError as err:
                continue
            for key in var.keys(): # Check data for masks
                var[key] = checkMask(var[key], key + ' (in ' + varName + ')')
            fieldpacket = makeFieldpacket(varName, var, isLocal)
            grid = 'Local' if isLocal else 'Regional'
            kwargs = {}
            if varName == 'waveHs':
                waveDm = gm.getStwaveField('waveDm', prefix, isLocal, model=model)['waveDm']
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
    
    print('Total runtime = ' + str(DT.datetime.now() - start))

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
    inp = sys.argv[1:]
    # Make sure arguments are passed correctly
    if len(inp) != 4 and len(inp) == 0:
        sys.exit('Error: four arguments are necessary. ' +
                'Please refer to help (python dailyPlots.py help)')
    # Help message
    if inp[0] == 'help':
        print('----------------------------------------------')
        print('Usage is as follows:')
        print('python dailyPlots.py endTime numDays model prefix workingDirectory')
        print('----------------------------------------------')
        print('endTime is the ending datetime in the format like 2018-04-01T15:00:00Z')
        print('numDays is the number of days (integer) prior to endTime to use')
        print('model is one of: STWAVE, CMS')
        print('prefix is one of: HP, FP, CB')
        print('workingDirectory is the path where the plots will be saved')
        sys.exit()
    # End and start times
    try:
        endTime = DT.datetime.strptime(inp[0], '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        sys.exit('Error: the first argument needs to be a datetime in the ' +
                'format %Y-%m-%dT%H:%M:%SZ')
    try:
        numDays = int(inp[1])
    except ValueError:
        sys.exit('Error: the second argument needs to be an integer')
    model = inp[2]
    if model not in ['STWAVE', 'CMS']:
        sys.exit('Error: the third argument needs to be one of: STWAVE, CMS')
    prefix = inp[3]
    if prefix not in ['HP', 'FP', 'CB']:
        sys.exit('Error: the third argument needs to be one of: HP, FP')
    startTime = endTime - DT.timedelta(days=numDays)
    # Working directory
    workingDirectory = inp[4]
    return startTime, endTime, model, prefix, workingDirectory

def getStats(startTime, endTime, model, prefix, station, go, gm):
    # Get observation data
    wo = go.getWaveSpec(station)
    if 'time' not in wo:
        return None, None, None, None, None, None
    if station in go.directional: # directional spectra
        if prefix is not 'FP': # half plane
            wo['dWED'], wo['wavedirbin'] = sbwave.HPchop_spec(wo['dWED'], 
            wo['wavedirbin'], angadj=angadj)
        obsStats = sbwave.waveStat(wo['dWED'], wo['wavefreqbin'], wo['wavedirbin'])
    else: # non-directional
        obsStats = sbwave.stats1D(wo['fspec'], wo['wavefreqbin'])
    # Get model results
    with warnings.catch_warnings(record=True) as w:
        wm = gm.getWaveSpecSTWAVE(prefix, station, model=model)
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