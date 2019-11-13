# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, re
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback import frontBackCMS
from getdatatestbed import getDataFRF
from testbedutils import fileHandling
from prepdata import inputOutput


def Master_CMS_run(inputDict):
    """

    Args:
        inputDict:

    Returns:

    """

    # first up, need to check which parts I am running
    waveFlag = inputDict.get('wave', True)
    flowFlag = inputDict.get('flow', False)
    morphFlag = inputDict.get('morph', False)
    # parse out the rest of the input dictionary
    endTime_str = inputDict['endTime']
    startTime_str = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['pFlag']
    version_prefix = inputDict.get('version_prefix', 'base')
    model = inputDict.get('model', 'CMS').lower()
    fileHandling.checkVersionPrefix(model, version_prefix)

    # __________________input directories________________________________
    codeDir = os.getcwd()                                # location of root cmtb directory
    # check executable
    if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
        inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])
    inputDict['path_prefix'] = os.path.join(workingDir, model, version_prefix)

    # ______________________ Logging  ____________________________
    LOG_FILENAME = fileHandling.logFileLogic(inputDict['path_prefix'], version_prefix, startTime_str.replace(':', ''),
                              endTime_str.replace(':', ''), log=False)
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    try:
        projectEnd = DT.datetime.strptime(endTime_str, '%Y-%m-%dT%H:%M:%SZ')
        projectStart = DT.datetime.strptime(startTime_str, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        assert len(endTime_str) == 10, 'Your Time does not fit convention, check T/Z and input format'

    # check the surveyNumber of the previous days run
    cmtb_data = getDataFRF.getDataTestBed(projectStart, projectEnd, inputDict['THREDDS'])
    go = getDataFRF.getObs(projectStart-DT.timedelta(days=simulationDuration), projectEnd)  # add pad to make sure there's enough data for coldStart
    # now get all Data
    allBathyTime = cmtb_data.getBathyIntegratedTransect(xbounds=[945, 950], ybounds=[945, 950], forceReturnAll=True)['time']  # just for time
    allWaves = go.getWaveSpec('waverider-26m')
    allWinds = go.getWind()
    allWL = go.getWL()

    # TODO can i just pull all cold start dates and pass to next section to make decisions???

    # # try to pull the .nc file of the previous run. -> this code is requried ONLY if we want to hot start CMS Flow!
    # try:
    #     ## this section checks to see if i need to re-run simulations that were previously run with old bathymetry (identifying cold starts)
    #     timeYesterday = projectStart - DT.timedelta(days=1)  # find yesterdays simulation in datetime
    #     cmsIO_yesterday = inputOutput.cmsfIO(path=os.path.join(inputDict['path_prefix'], DT.datetime.strftime(timeYesterday, '%Y-%m-%dT%H%M%SZ')))               # initialize the class
    #     # get into the directory I need
    #     cmsIO_yesterday.read_CMSF_all()
    #     cmsIO_yesterday.read_CMSF_telnc()
    #
    #     # what survey number did this thing use??
    #     prev_mod_stime = nc.num2date(cmsIO_yesterday.telnc_dict['surveyTime'][0], units='seconds since 1970-01-01')
    #     CSflag = False
    #
    #     # what time was this survey number?  this says that if
    #     # 1 - the previous model used a survey older than the latest survey
    #     # 2 - the previous model started AFTER the latest survey (i.e., it should have used the latest survey)
    #     if (allBathy > prev_mod_stime) and (timeYesterday > allBathy):
    #         d1_N = allBathy.replace(microsecond=0, second=0, minute=0, hour=0)
    #         if d1_N != allBathy:
    #             # this means we rounded it down and have to add back a day to start on the 00:00:00 after the survey
    #             d1_N = d1_N + DT.timedelta(days=1)
    #         # reset the first day of the simulations to be the day after or of the latest survey
    #         # (depending on if the survey time is 00:00:00 or 12:00:00)
    #         projectStart = d1_N
    #         CSflag = True
    #
    # except (IOError, OSError):
    #     # this means that this is the first time this has been run, so you MUST coldstart
    #     CSflag = True

    # This is the portion that creates a list of simulation endTimes
    simDur_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    dateStartList = [projectStart]
    dateStringList = [dateStartList[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((projectEnd - projectStart).total_seconds()/simDur_DT.total_seconds()))-1):
        dateStartList.append(dateStartList[-1] + simDur_DT)
        dateStringList.append(dateStartList[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, model)

    # ________________________________________________ RUNNING LOOP ________________________________________________
    errors, errorDates, curdir = [], [], os.getcwd()
    for time in dateStringList:
        workingDirectory = os.path.join(inputDict['path_prefix'],
                                        ''.join(time.split(':')))  # moving to the new simulation's folder
        try:
            if generateFlag == True:
                # d1Flow, d2, cmsfio
                startTime, d2, cmsfio = frontBackCMS.CMSFsimSetup(time, inputDict=inputDict, bathyTimes=allBathyTime, allWL=allWL, allWaves=allWaves, allWind=allWinds)
                print('check Time input to cmsWave is that to match up cms Flow')
                frontBackCMS.CMSsimSetup(time, inputDict=inputDict, bathyTimes=allBathyTime, allWL=allWL, allWaves=allWaves, allWind=allWinds)

            if runFlag == True: # run model
                os.chdir(workingDirectory)  # changing locations to where input files should be made
                dt = DT.datetime.now()
                print('Beginning {} Simulation {}'.format(model, dt))

                if waveFlag is True or flowFlag is True:
                    _ = check_output(os.path.join(codeDir, inputDict['modelExecutable'] + ' {}.sim'.format(''.join(time.split(':')))), shell=True)

                print('Simulation took %s ' % (DT.datetime.now() - dt))
                os.chdir(curdir)

            if analyzeFlag == True:
                print('**\nBegin Analyze Script %s ' % DT.datetime.now())
                if waveFlag:
                    frontBackCMS.CMSanalyze(time, inputDict=inputDict)

            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/cmtb')
                    print('moved %s ' % file)

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP : {}'.format(e))
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)
            os.chdir(curdir)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : CMS Wave and Flow')

    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.load(f)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/CMS_Input_example for example yaml file.')

    Master_CMS_run(inputDict=inputDict)
