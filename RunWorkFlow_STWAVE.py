# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackSTWAVE import STanalyze, STsimSetup
import os, getopt, sys, shutil, glob, platform, logging, yaml
from testbedutils import fileHandling
from getdatatestbed import getDataFRF

def Master_STWAVE_run(inputDict):
    """This will run STWAVE with any version prefix given start, end, and timestep
    `   version_prefix, startTime, endTime, simulationDuration, all of which are in the input dictionary loaded from the
    input yaml file
    This is the generic work flow

    Args:
      inputDict:
        key pFlag: plots or not (boolean)
        :key analyzeFlag: analyze results or not (boolean)
        :key generateFlag: generate input files or not (boolean)
        :key runFlag: run the simulation or not (boolean)
        :key start_date: date I am starting the simulation (format '2018-01-15T00:00:00Z')
        :key end_date: date I am ending the simulation (format '2018-01-15T00:00:00Z')
        :key workingDirectory: path to the working directory the user wants
        :key netCDFdir: path to the netCDF save location specified by the user
        :key THREDDS: which THREDDS server are we using, 'FRF' or 'CHL'
        :key version_prefix: right now we have 'FIXED', 'MOBILE', or 'MOBILE_RESET'
        :key duration: how long you want the simulations to run in hours (24 by default)

    Returns:
      None

    """
    ###################################################################################################################
    #######################   Parse out input Dictionary     ##########################################################
    ###################################################################################################################
    # required inputs
    model = inputDict['modelSettings'].get('model', 'STWAVE').lower()
    hostfile = inputDict.get('hostfileLoc', 'hostFile-IB')
    version_prefix = inputDict['modelSettings']['version_prefix']
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict.get('simulationDuration', 24)
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    runNested = inputDict['modelSettings'].get('runNested', True)
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['plotFlag']
    FRFgaugelocsFile = inputDict.get('sensorLocPkl', 'ArchiveFolder/frf_sensor_locations.pkl')


    ###################################################################################################################
    #######################   doing Data check and setting up input vars  #############################################
    ###################################################################################################################

    # __________________input vars________________________________
    executableLocation = inputDict['modelExecutable']
    # __________________pre-processing checks________________________________
    fileHandling.checkVersionPrefix(model, inputDict)
    # __________________input directories________________________________
    ## handle Architecture here
    if 'ForcedSurveyDate' in list(inputDict.keys()):
        workingDirectory = os.path.join(workingDir, model.lower(), version_prefix, 'SingleBathy_{}'.format(
                                                                        inputDict['modelSettings']['ForcedSurveyDate']))
        ForcedSurveyDate =  inputDict['modelSettings']['ForcedSurveyDate']
    else:
        workingDirectory = os.path.join(workingDir, 'waveModels', model.lower(), version_prefix)
        ForcedSurveyDate = None
    inputDict['path_prefix'] = workingDirectory
    # ______________________ Logging  ____________________________
    LOG_FILENAME = fileHandling.logFileLogic(workingDirectory, version_prefix, startTime, endTime, log=False)

    # __________________get time list to loop over________________________________
    dateStartList, dateStringList, projectStart, projectEnd  = fileHandling.createTimeInfo(startTime, endTime,
                                                                                           simulationDuration)
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, model)

    # ______________________________gather all data _____________________________
    if generateFlag == True:
        go = getDataFRF.getObs(projectStart, projectEnd)  # initialize get observation
        rawspec = go.getWaveSpec(gaugenumber='waverider-26m', specOnly=True)
        rawWL = go.getWL()
        rawwind = go.getWind(gaugenumber=0)
        loc_dict = go.get_sensor_locations(datafile=FRFgaugelocsFile, window_days=14)
        gtb = getDataFRF.getDataTestBed(projectStart, projectEnd)  # this should be relocated to operational servers
        bathy = gtb.getBathyIntegratedTransect(method=1, ForcedSurveyDate=ForcedSurveyDate)

    # ________________________________________________ RUN LOOP ________________________________________________
    # run the process through each of the above dates
    errors, errorDates, curdir = [], [], os.getcwd()
    for time in dateStringList:
        print(' ------------------------------ START %s --------------------------------' %time)

        try:
            datadir = os.path.join(workingDirectory, ''.join(time.split(':')))  # moving to the new simulation's folder
            if generateFlag == True:
                [nproc_par, nproc_nest] = STsimSetup(time, inputDict, rawwind, rawWL, rawspec, bathy, loc_dict)

                if nproc_par == -1 or nproc_nest == -1:
                    print('************************\nNo Data available\naborting run\n***********************')
                    # remove generated files?
                    shutil.rmtree(os.path.join(workingDirectory,''.join(time.split(':'))))
                    continue  # this is to return to the next time step if there's no data

            if runFlag == True:
                os.chdir(datadir)  # changing locations to where simulation files live
                t= DT.datetime.now()
                print('Beggining Parent Simulation %s' %t)
                try:
                    assert os.path.isfile(hostfile), 'Check hostfile path'
                    parent = check_output('mpiexec -n {} -f {} {} {}.sim'.format(nproc_par, hostfile, executableLocation, ''.join(time.split(':'))), shell=True)
                except AssertionError:
                    import multiprocessing
                    count = multiprocessing.cpu_count()  # Max out computer cores
                    parent = check_output('mpiexec -n {} {} {}.sim'.format(count, executableLocation, ''.join(time.split(':'))), shell=True)
                if runNested is not False:
                    try:
                        assert os.path.isfile(hostfile), 'Check hostfile path'
                        child = check_output('mpiexec -n {} -f {} {} {}nested.sim'.format(nproc_nest, hostfile, executableLocation, ''.join(time.split(':'))), shell=True)
                    except AssertionError:
                        import multiprocessing
                        count = multiprocessing.cpu_count()
                        if count > nproc_nest:
                            count = nproc_nest # lower the processors called for to match sim file (otherwise will throw segfault)
                        child = check_output('mpiexec -n {} {} {}nested.sim'.format(count, executableLocation, ''.join(time.split(':'))), shell=True)
                print(('  Simulations took {:.2f} hours'.format((DT.datetime.now() - t).total_seconds()/3600)))
            # run analyze and archive script
            os.chdir(curdir)  # change back after runing simulation locally
            if analyzeFlag == True:
                beachWaves = STanalyze(time, inputDict)
            if pFlag == True and DT.date.today() == endTime.date():
                print('**\n Moving Plots! \n &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                # move files
                moveFnames = glob.glob(datadir + '/figures/CMTB*.png')
                moveFnames.extend(glob.glob(datadir + '/figures/CMTB*.gif'))
                for file in moveFnames:
                    shutil.copy(file,  '/mnt/gaia/gages/results/frfIn/CMTB')
                    print('moved %s ' % file)
            print(' --------------   SUCCESS: Done %s --------------------------------' %time)
        except Exception as e:
            os.chdir(curdir)  # if things break during run flag, need to get back out!
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            # print e
            print(e.args)
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : STWAVE')
    import yaml
    yamlLoc = args[0]
    try:
        yamlLoc = args[0]
        if os.path.exists('.cmtbSettings'):
            with open('.cmtbSettings', 'r') as fid:
                a = yaml.safe_load(fid)
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.safe_load(f)
        inputDict.update(a)
    except:
        raise IOError('Input Yaml required. see yaml_files/TestBedExampleInputs/STWAVE_Input_example.yml for example')

    # run work flow
    Master_STWAVE_run(inputDict)
