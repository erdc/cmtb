"""
This script holds the master function for the simulation Setup
for the Swash model setup
"""
import testbedutils.anglesLib
from prepdata import inputOutput
from prepdata.prepDataLib import PrepDataTools as STPD
from getdatatestbed.getDataFRF import getDataTestBed
from getdatatestbed.getDataFRF import getObs
import datetime as DT
import os, glob, makenc, pickle, tarfile
import netCDF4 as nc
import numpy as np
from prepdata import prepDataLib as STPD
from prepdata.inputOutput import swashIO
from getdatatestbed import getDataFRF
import plotting.operationalPlots as oP
from testbedutils import sblib as sb
from testbedutils import waveLib as sbwave
from plotting.operationalPlots import obs_V_mod_TS
from testbedutils import geoprocess as gp


def SwashSimSetup(startTime, inputDict):
    """This Function is the master call for the  data preparation for the Coastal Model
    Test Bed (CMTB) and the Swash wave/FLow model


    NOTE: input to the function is the end of the duration.  All Files are labeled by this convention
    all time stamps otherwise are top of the data collection

    Args:
        startTime (str): this is a string of format YYYY-mm-ddTHH:MM:SSZ (or YYYY-mm-dd) in UTC time
        inputDict (dict): this is a dictionary that is read from the yaml read function

    """
    # begin by setting up input parameters
    timerun = inputDict.get('simulationDuration', 24)
    pFlag = inputDict.get('pFlag', True)
    server = inputDict.get('THREDDS', 'CHL')
    # this raises error if not present (intended)
    version_prefix = inputDict['version_prefix']
    path_prefix = inputDict['path_prefix']  # data super directiory
    # ______________________________________________________________________________
    # define version parameters
    versionlist = ['base', 'ts']
    assert version_prefix.lower() in versionlist, 'Please check your version Prefix'
    # here is where we set something that would handle 3D mode or time series mode,
    # might set flags for preprocessing below
    # _______________________________________________________________________________
    # set times
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, timerun * 3600, 0)
    date_str = d1.strftime('%Y-%m-%dT%H%M%SZ')  # used to be endtime

    if isinstance(timerun, str):
        timerun = int(timerun)

    # __________________Make Working Data Directories_____________________________________________
    #
    if not os.path.exists(os.path.join(path_prefix, date_str)):  # if it doesn't exist
        os.makedirs(os.path.join(path_prefix, date_str))  # make the directory
    if not os.path.exists(os.path.join(path_prefix, date_str)):
        os.makedirs(os.path.join(path_prefix, date_str))

    print("Model Time Start : %s  Model Time End:  %s" % (d1, d2))
    print("OPERATIONAL files will be place in {} folder".format(os.path.join(path_prefix, date_str)))
    # ______________________________________________________________________________
    # begin model data gathering
    go = getObs(d1, d2, THREDDS=server)                  # initialize get observation class
    prepdata = STPD.PrepDataTools()                      # for preprocessing
    gdTB = getDataTestBed(d1, d2, THREDDS=server)        # for bathy data gathering
    # _____________WAVES____________________________
    print('_________________\nGetting Wave Data')
    rawspec = go.getWaveSpec(gaugenumber= '8m-array') #'AWAC-11m')  #
    assert 'time' in rawspec, "\n++++\nThere's No Wave data between %s and %s \n++++\n" % (d1, d2)
    # preprocess wave spectra
    if version_prefix.lower() == 'base':
        wavepacket = prepdata.prep_SWASH_spec(rawspec, version_prefix)
    else:
        raise NotImplementedError('pre-process TS data ')

    # _____________WINDS______________________
    print('_________________\nSkipping Wind')
    # try:
    #     rawwind = go.getWind(gaugenumber=0)
    #     # average and rotate winds
    #     windpacket = prepdata.prep_wind(rawwind, wavepacket['epochtime'])
    #     # wind height correction
    #     print('number of wind records %d with %d interpolated points' % (
    #         np.size(windpacket['time']), sum(windpacket['flag'])))
    # except (RuntimeError, TypeError):
    #     windpacket = None
    #     print(' NO WIND ON RECORD')

    ## ___________WATER LEVEL__________________
    print('_________________\nGetting Water Level Data')
    try:
        # get water level data
        rawWL = go.getWL()
        # average WL
        WLpacket = prepdata.prep_WL(rawWL, wavepacket['epochtime'])
        print('number of WL records %d, with %d interpolated points' % (
            np.size(WLpacket['time']), sum(WLpacket['flag'])))
    except (RuntimeError, TypeError):
        WLpacket = None
    ### ____________ Get bathy grid from thredds ________________
    # bathy = gdTB.getGridSwash(method='historical')
    bathy = gdTB.getBathyIntegratedTransect(method=1, ybound=[940, 950])
    swsinfo, gridDict = prepdata.prep_SwashBathy(wavepacket['xFRF'], wavepacket['yFRF'], bathy, dx=1, dy=1,
                                                 yBounds=[944, 947])  # non-inclusive index if you want 3 make 4 wide

    ## begin output
    # set some of the class instance variables before writing Sws file
    swio = swashIO(WL=WLpacket['avgWL'], equilbTime=wavepacket['spinUp'],
                   Hs=wavepacket['Hs'], Tp=1/wavepacket['peakf'],
                   Dm=wavepacket['waveDm'], ofileNameBase=date_str,
                   path_prefix=path_prefix, version_prefix=version_prefix,
                   nprocess=gridDict['h'].shape[0])

    # write SWS file first
    swio.write_sws(swsinfo)
    swio.write_spec1D(wavepacket['freqbins'], wavepacket['fspec'])
    swio.write_bot(gridDict['h'])
    # now write QA/
    print(' TODO: figure OUt how to write flags, and what flags to write!!')
    swio.flags = None
    pickleName = os.path.join(path_prefix, date_str,'.pickle')
    with open(pickleName, 'wb') as fid:
        pickle.dump(swio, fid, protocol=pickle.HIGHEST_PROTOCOL)
    return swio

def SwashAnalyze(startTime, inputDict, swio):
    """This runs the post process script for Swash wave will create plots and netcdf files at request

    Args:
        inputDict (dict): this is an input dictionary that was generated with the
            keys from the project input yaml file
        startTime (str): input start time with datestring in format YYYY-mm-ddThh:mm:ssZ


    Returns:
        plots in the inputDict['workingDirectory'] location
        netCDF files to the inputDict['netCDFdir'] directory

    """
    print("check docstrings for Analyze and preprocess")
    # ___________________define Global Variables___________________________________

    pFlag = inputDict.get('pFlag', True)
    version_prefix = inputDict['version_prefix']
    Thredds_Base = inputDict.get('netCDFdir', '/thredds_data')
    server = inputDict.get('THREDDS', 'CHL')
    # the below should error if not included in input Dict
    path_prefix = inputDict['path_prefix']  # for organizing data
    simulationDuration = inputDict['simulationDuration']
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # establishing the resolution of the input datetime
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    d2 = d1 + DT.timedelta(0, simulationDuration * 3600, 0)
    datestring = d1.strftime('%Y-%m-%dT%H%M%SZ')  # a string for file names
    fpath = os.path.join(path_prefix, datestring)
    model = 'Swash'
    # ____________________________________________________________________________
    #________________________________________________________

    print('\nBeggining of Analyze Script\nLooking for file in ' + fpath)
    print('\nData Start: %s  Finish: %s' % (d1, d2))
    go = getDataFRF.getObs(d1, d2, server)  # setting up get data instance
    prepdata = STPD.PrepDataTools()  # initializing instance for rotation scheme
    swio = swio                             # initializing read/write class as passed (has previous info from setup)

    ######################################################################################################################
    ######################################################################################################################
    ##################################   Load Data Here / Massage Data Here   ############################################
    ######################################################################################################################
    ######################################################################################################################
    matfile = os.path.join(swio.path_prefix, ''.join(swio.ofileNameBase.split('-'))+'.mat')
    print(' TODO: write run wall time to output file')
    print('Loading files ')
    simData, simMeta = swio.loadSwash_Mat(fname=matfile)  # load all files
    ######################################################################################################################
    ######################################################################################################################
    ##################################  plotting #########################################################################
    ######################################################################################################################
    ######################################################################################################################

    ## do some plotting

    if not os.path.exists(os.path.join(path_prefix,datestring, 'figures')):
        os.makedirs(os.path.join(path_prefix,datestring, 'figures'))  # make the directory for the simulation plots
    figureBaseFname = 'CMTB_waveModels_{}_{}_'.format(model, version_prefix)
    from matplotlib import pyplot as plt
    # plt.figure()
    # plt.plot()
    #
    # plot time-series of cross-shore evolution movie
    print('this script makes double pictures, we can probably sub-sample')

    for tidx, timeStep in enumerate(simData['time']):
        ofPlotName = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_' + timeStep.strftime('%Y%m%dT%H%M%S%fZ') +'.png')
        oP.generate_CrossShoreTimeseries(ofPlotName, simData['eta'][tidx].squeeze(), -simData['elevation'], simData['xFRF'])
    ## now make gif of waves moving across shore
    imgList = sorted(glob.glob(os.path.join(path_prefix, datestring, 'figures', '*_TS_*.png')))
    sb.makegif(imgList, os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS_{}.gif'.format(datestring)),dt=0.1)

    tarOutFile  = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TS.tar.gz')
    sb.myTarMaker(tarOutFile, imgList)
    [os.remove(ff) for ff in imgList]


    ofname = os.path.join(path_prefix, datestring, 'figures', figureBaseFname + 'TimeStack.png')
    ## figure making
    cmap = 'RdBu'
    plt.figure()
    mappable = plt.pcolor(simData['xFRF'], simData['time'], simData['eta'].squeeze(), cmap=cmap)
    plt.colorbar(mappable)
    plt.title('Time Stack for {} transect'.format(simData['yFRF']))
    plt.ylabel('cross-shore position')
    plt.xlabel('simulation time(s)')
    plt.savefig(ofname)
    plt.close()

    # ################################
    #        Make NETCDF files       #
    # ################################
    ## before netCDF.
    # get significant wave height for cross shore
    # slice the time series so we're on ly isolating the non-repeating time series of data
    #################################################3
    fldrArch = os.path.join(model, version_prefix)
    spatial = {'time': nc.date2num(wave_pack['time'], units='seconds since 1970-01-01 00:00:00'),
               'station_name': 'Regional Simulation Field Data',
               'waveHs': np.transpose(wave_pack['waveHs'], (0, 2, 1)),  # put into dimensions [t, y, x]
               'waveTm': np.transpose(np.ones_like(wave_pack['waveHs']) * -999, (0, 2, 1)),
               'waveDm': np.transpose(wave_pack['waveDm'], (0, 2, 1)),  # put into dimensions [t, y, x]
               'waveTp': np.transpose(wave_pack['waveTp'], (0, 2, 1)),  # put into dimensions [t, y, x]
               'bathymetry': np.transpose(gridPack['elevation'], (0, 2, 1)),  # put into dimensions [t, y, x]
               'latitude': gridPack['latitude'],  # put into dimensions [t, y, x] - NOT WRITTEN TO FILE
               'longitude': gridPack['longitude'],  # put into dimensions [t, y, x] - NOT WRITTEN TO FILE
               'xFRF': gridPack['xFRF'],  # put into dimensions [t, y, x]
               'yFRF': gridPack['yFRF'],  # put into dimensions [t, y, x]
               ######################
               'DX': dep_pack['dx'],
               'DX': dep_pack['dy'],
               'NI': dep_pack['NI'],
               'NJ': dep_pack['NJ'],
               'grid_azimuth': gridPack['azimuth']
               }

    TdsFldrBase = os.path.join(Thredds_Base, fldrArch)
    NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, 'Field'), datestring, model=model)
    # make the name of this nc file
    NCname = 'CMTB-waveModels_{}_{}_Field_{}.nc'.format(model, version_prefix, datestring)
    fieldOfname = os.path.join(NCpath,
                               NCname)  # TdsFldrBase + '/CMTB-waveModels_Swash_{}_Local-Field_%s.nc'.format(version_prefix, datestring)

    if not os.path.exists(TdsFldrBase):
        os.makedirs(TdsFldrBase)  # make the directory for the thredds data output
    if not os.path.exists(os.path.join(TdsFldrBase, 'Field', 'Field.ncml')):
        inputOutput.makencml(os.path.join(TdsFldrBase, 'Field', 'Field.ncml'))  # remake the ncml if its not there
    # make file name strings
    flagfname = os.path.join(fpath, 'Flags{}.out.txt'.format(datestring))  # startTime # the name of flag file
    fieldYaml = 'yaml_files/waveModels/%s/Field_globalmeta.yml' % (fldrArch)  # field
    varYaml = 'yaml_files/waveModels/%s/Field_var.yml' % (fldrArch)
    assert os.path.isfile(fieldYaml), 'NetCDF yaml files are not created'  # make sure yaml file is in place
    makenc.makenc_field(data_lib=spatial, globalyaml_fname=fieldYaml, flagfname=flagfname,
                        ofname=fieldOfname, var_yaml_fname=varYaml)
    ###################################################################################################################
    ###############################   Plotting  Below   ###############################################################
    ###################################################################################################################
    dep_pack['bathy'] = np.transpose(dep_pack['bathy'], (0, 2, 1))  # dims [t, y, x]
    plotParams = [('waveHs', '$m$'), ('bathymetry', 'NAVD88 $[m]$'), ('waveTp', '$s$'), ('waveDm', '$degTn$')]
    if pFlag == True:
        for param in plotParams:
            print('    plotting %s...' % param[0])
            spatialPlotPack = {'title': 'Regional Grid: %s' % param[0],
                               'xlabel': 'Longshore distance [m]',
                               'ylabel': 'Cross-shore distance [m]',
                               'field': spatial[param[0]],
                               'xcoord': spatial['xFRF'],
                               'ycoord': spatial['yFRF'],
                               'cblabel': '%s - %s' % (param[0], param[1]),
                               'time': nc.num2date(spatial['time'], 'seconds since 1970-01-01')}
            fnameSuffix = 'figures/CMTB_Swash_%s_%s' % (version_prefix, param[0])
            if param[0] == 'waveHs':
                oP.plotSpatialFieldData(dep_pack, spatialPlotPack, os.path.join(fpath, fnameSuffix), nested=0, directions=spatial['waveDm'])
            else:
                oP.plotSpatialFieldData(dep_pack, spatialPlotPack, os.path.join(fpath, fnameSuffix), nested=0)
            # now make a gif for each one, then delete pictures
            fList = sorted(glob.glob(fpath + '/figures/*%s*.png' % param[0]))
            sb.makegif(fList, fpath + '/figures/CMTB_%s_%s_%s.gif' % (version_prefix, param[0], datestring))
            [os.remove(ff) for ff in fList]

    ######################################################################################################################
    ######################################################################################################################
    ##################################  Wave Station Files HERE (loop) ###################################################
    ######################################################################################################################
    ######################################################################################################################

    # this is a list of file names to be made with station data from the parent simulation
    stationList = ['waverider-26m', 'waverider-17m', 'awac-11m', '8m-array', 'awac-6m', 'awac-4.5m', 'adop-3.5m',
                   'xp200m', 'xp150m', 'xp125m']
    for gg, station in enumerate(stationList):
        # stationName = 'CMTB-waveModels_Swash_%s_%s' % (version_prefix, station)  # xp 125

        # this needs to be the same order as the run script
        stat_yaml_fname = 'yaml_files/waveModels/{}/Station_var.yml'.format(fldrArch)
        globalyaml_fname = 'yaml_files/waveModels/{}/Station_globalmeta.yml'.format(fldrArch)
        # getting lat lon, easting northing idx's
        # Idx_i = len(gridPack['i']) - np.argwhere(gridPack['i'] == stat_packet['iStation'][
        #     gg]).squeeze() - 1  # to invert the coordinates from offshore 0 to onshore 0
        # Idx_j = np.argwhere(gridPack['j'] == stat_packet['jStation'][gg]).squeeze()
        if pFlag == True:
            w = go.getWaveSpec(station)  # go get all data
        else:
            w = go.getWaveGaugeLoc(station)
        # print '   Comparison location taken from thredds, check positioning '
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
                     'station_name': station,
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

        NCpath = sb.makeNCdir(Thredds_Base, os.path.join(version_prefix, station), datestring, model='Swash')
        # make the name of this nc file
        NCname = 'CMTB-waveModels_{}_{}_{}_{}.nc'.format(model, version_prefix, station, datestring)
        outFileName = os.path.join(NCpath, NCname)

        if not os.path.exists(TdsFldrBase):
            os.makedirs(TdsFldrBase)  # make the directory for the file/ncml to go into
        if not os.path.exists(os.path.join(TdsFldrBase, station + '.ncml')):
            inputOutput.makencml(os.path.join(TdsFldrBase, station + '.ncml'))
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

            timeStep, obsi, modi = sb.timeMatch(nc.date2num(w['time'], 'seconds since 1970-01-01'),
                                            np.arange(w['time'].shape[0]),
                                            nc.date2num(stat_packet['time'][:], 'seconds since 1970-01-01'),
                                            np.arange(len(stat_packet['time'])))  # time match

            for param in modStats:  # loop through each bulk statistic
                if len(timeStep) > 1 and param in ['Hm0', 'Tm', 'sprdF', 'sprdD', 'Tp', 'Dm']:
                    print('    plotting %s: %s' % (station, param))
                    if param in ['Tp', 'Tm10']:
                        units = 's'
                        title = '%s period' % param
                    elif param in ['Hm0']:
                        units = 'm'
                        title = 'Wave Height %s ' % param
                    elif param in ['Dm', 'Dp']:
                        units = 'degrees'
                        title = 'Direction %s' % param
                    elif param in ['sprdF', 'sprdD']:
                        units = '_.'
                        title = 'Spread %s ' % param

                    # now run plots
                    p_dict = {'time': nc.num2date(timeStep, 'seconds since 1970-01-01'),
                              'obs': obsStats[param][obsi.astype(int)],
                              'model': modStats[param][modi.astype(int)],
                              'var_name': param,
                              'units': units,  # ) -> this will be put inside a tex math environment!!!!
                              'p_title': title}

                    ofname = os.path.join(fpath, 'figures/Station_%s_%s_%s.png' % (station, param, datestring))
                    stats = obs_V_mod_TS(ofname, p_dict, logo_path='ArchiveFolder/CHL_logo.png')

                    if station == 'waverider-26m' and param == 'Hm0':
                        # this is a fail safe to abort run if the boundary conditions don't
                        # meet quality standards below
                        bias = 0.1  # bias has to be within 10 centimeters
                        RMSE = 0.1  # RMSE has to be within 10 centimeters
                        if isinstance(p_dict['obs'], np.ma.masked_array) and ~p_dict['obs'].mask.any():
                            p_dict['obs'] = np.array(p_dict['obs'])
                        # try:
                        #     # assert stats['RMSE'] < RMSE, 'RMSE test on spectral boundary energy failed'
                        #     # assert np.abs(stats['bias']) < bias, 'bias test on spectral boundary energy failed'
                        # except:
                        #     print '!!!!!!!!!!FAILED BOUNDARY!!!!!!!!'
                        #     print 'deleting data from thredds!'
                        #     os.remove(fieldOfname)
                        #     os.remove(outFileName)
                        #     raise RuntimeError('The Model Is not validating its offshore boundary condition')
