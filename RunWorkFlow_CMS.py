# -*- coding: utf-8 -*-
#!/home/number/anaconda2/bin/python
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackCMS import CMSanalyze, CMSFanalyze
from frontback.frontBackCMS import CMSsimSetup
from getdatatestbed.getDataFRF import getDataTestBed
from prepdata import inputOutput
import netCDF4 as nc


def Master_CMS_run(inputDict):
    '''
    This function will run CMS with any version prefix given start, end, and timestep

    :param inputDict: a dictionary that is read from the input yaml
    :return: implementation of the CMTB CMS as the user has specified in the inputDict.
    '''


    # first up, need to check which parts I am running
    waveFlag = inputDict['wave']
    flowFlag = inputDict['flow']
    morphFlag = inputDict['morph']

    version_prefix = 'CMS'
    if waveFlag:
        wave_version_prefix = inputDict['wave_version_prefix']
        version_prefix = version_prefix + '_' + wave_version_prefix
        # data check
        prefixList = np.array(['HP', 'UNTUNED'])
        assert (wave_version_prefix == prefixList).any(), "Please enter a valid wave version prefix\n Prefix assigned = %s must be in List %s" % (wave_version_prefix, prefixList)
    if flowFlag:
        flow_version_prefix = inputDict['flow_version_prefix']
        version_prefix = version_prefix + '_' + flow_version_prefix
        # data check
        prefixList = np.array(['STANDARD'])
        assert (flow_version_prefix == prefixList).any(), "Please enter a valid flow version prefix\n Prefix assigned = %s must be in List %s" % (flow_version_prefix, prefixList)
    if morphFlag:
        morph_version_prefix = inputDict['morph_version_prefix']
        version_prefix = version_prefix + '_' + morph_version_prefix
        # data check
        prefixList = np.array(['FIXED', 'MOBILE', 'MOBILE_RESET'])
        assert (morph_version_prefix == prefixList).any(), "Please enter a valid morph version prefix\n Prefix assigned = %s must be in List %s" % (morph_version_prefix, prefixList)

    endTime = inputDict['end_date']
    startTime = inputDict['start_date']
    simulationDuration = inputDict['duration']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['pFlag']


    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    if inputDict['waveExecutable'].startswith(codeDir):  # change to relative path
        import re
        inputDict['waveExecutable'] = re.sub(codeDir, '', inputDict['waveExecutable'])
    # check executable
    if inputDict['flowExecutable'].startswith(codeDir):  # change to relative path
        import re
        inputDict['flowExecutable'] = re.sub(codeDir, '', inputDict['flowExecutable'])

    if workingDir[-1] == '/':
        outDataBase = workingDir + 'CMS/' + version_prefix + '/'  #codeDir + '/%s_CSHORE_data/' % version_prefix
    else:
        outDataBase = workingDir + '/CMS/' + version_prefix +'/'

    inputDict['path_prefix'] = outDataBase
    TOD = 0  # 0=start simulations at 0000
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end time?
    LOG_FILENAME = inputDict['logfileLoc'] + '/CMS/%s/logs/CMTB_BatchRun_Log_%s_%s_%s.log' % (version_prefix, version_prefix, startTime.replace(':', ''), endTime.replace(':', ''))

    try:
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    except IOError:
        os.makedirs(outDataBase+'logs')
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    logging.debug('\n-------------------\nTraceback Error Log for:\n\nSimulation Started: %s\n-------------------\n'
                  % (DT.datetime.now()))

    # ____________________________________________________________
    # establishing the resolution of the input datetime
    try:
        projectEnd = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
        projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
    except ValueError:
        assert len(endTime) == 10, 'Your Time does not fit convention, check T/Z and input format'

    # check the surveyNumber of the previous days run?
    cmtb_data = getDataTestBed(projectStart, projectStart + DT.timedelta(minutes=1), inputDict['THREDDS'])
    bathy_data = cmtb_data.getBathyIntegratedTransect()
    b_time = bathy_data['time']


    # try to pull the .nc file of the previous run.
    # -> this code is requried ONLY if we want to hot start CMS Flow!
    # as per discussion with SB on 12/0/3/2018 we are going to switch to always coldstart until the CMSF executable
    # has beeen fixed to hotstart correctly, so that we can continue to make progress.
    # try to pull the .nc file of the previous run.
    """
    CSflag = False
    try:
        Time_O = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') - DT.timedelta(days=1)
        # initialize the class
        cmsfIO_O = inputOutput.cmsfIO()
        # get into the directory I need
        cmsfIO_O.read_CMSF_all(outDataBase + DT.datetime.strftime(Time_O, '%Y-%m-%dT%H%M%SZ'))
        # what survey number did this thing use??
        prev_mod_stime = cmsfIO_O.telnc_dict['surveyTime'][0]
        # convert this to the datetime
        prev_mod_stime = nc.num2date(prev_mod_stime, units='seconds since 1970-01-01')

        # what time was this survey number
        # ok, this says that if
        # 1 - the previous model used a survey older than the latest survey
        # AND 2 - the previous model started AFTER the latest survey (i.e., it should have used the latest survey)
        if (b_time > prev_mod_stime) and (Time_O > b_time):
            d1_N = b_time.replace(microsecond=0, second=0, minute=0, hour=0)
            if d1_N == b_time:
                # this means that the survey already started on 00:00:00 of the day
                pass
            else:
                d1_N = d1_N + DT.timedelta(days=1)
                # this means we rounded it down and have to add back a day to start on the 00:00:00 after the survey
            # reset the first day of the simulations to be the day after or of the latest survey
            # (depending on if the survey time is 00:00:00 or 12:00:00)
            del projectStart
            projectStart = d1_N
            CSflag = True
            del d1_N
        else:
            pass
    except (IOError, OSError):
        # this means that this is the first time this has been run, so you MUST coldstart
        CSflag = True
    """


    # This is the portion that creates a list of simulation end times
    dt_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    dateStartList = [projectStart]
    dateStringList = [dateStartList[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((projectEnd-projectStart).total_seconds()/dt_DT.total_seconds()))-1):
        dateStartList.append(dateStartList[-1] + dt_DT)
        dateStringList.append(dateStartList[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))

    # toggle my cold start flags
    # -> this code is requried ONLY if we want to hot start CMS Flow!
    # as per discussion with SB on 12/0/3/2018 we are going to switch to always coldstart until the CMSF executable
    # has beeen fixed to hotstart correctly, so that we can continue to make progress.
    # try to pull the .nc file of the previous run.
    """
    csFlag = np.zeros(np.shape(np.array(dateStringList)), dtype=int)
    if CSflag:
        csFlag[0] = 1
    """
    csFlag = np.ones(np.shape(np.array(dateStringList)), dtype=int)

    errors, errorDates = [],[]
    curdir = os.getcwd()
    # ______________________________decide process and run _____________________________
    # run the process through each of the above dates
    print '\n-\n-\nMASTER WorkFLOW for CMS SIMULATIONS\n-\n-\n'
    print 'Batch Process Start: %s     Finish: %s '% (projectStart, projectEnd)
    print 'The batch simulation is Run in %s Version' % version_prefix
    print 'Check for simulation errors here %s' % LOG_FILENAME
    print '------------------------------------\n\n************************************\n\n------------------------------------\n\n'


    # ________________________________________________ RUNNING LOOP ________________________________________________
    cnt = 0
    for time in dateStringList:
        try:
            print '**\nBegin '
            print 'Beginning Simulation %s' %DT.datetime.now()

            # toggle my coldStart flags
            inputDict['csFlag'] = csFlag[cnt]
            cnt = cnt + 1

            if generateFlag == True:
                CMSsimSetup(time, inputDict=inputDict)
                datadir = outDataBase + ''.join(time.split(':'))  # moving to the new simulation's folder

            if runFlag == True: # run model
                os.chdir(datadir) # changing locations to where input files should be made
                print 'Running CMS Simulation'
                dt = DT.datetime.now()
                if waveFlag:
                    simOutput = check_output(codeDir + '%s %s.sim' %(inputDict['waveExecutable'], ''.join(time.split(':'))), shell=True)
                if flowFlag:
                    # copy over the executable
                    shutil.copy2(codeDir + '%s' %inputDict['flowExecutable'], datadir)
                    # copy over the .bid file
                    tempDir = os.getcwd().split('cmtb')
                    shutil.copy2(os.path.join(tempDir[0], 'cmtb/grids/CMS/CMS-Flow-FRF.bid',), datadir)
                    # rename this file
                    os.rename('CMS-Flow-FRF.bid', ''.join(time.split(':')) + '.bid')

                    # show time!
                    simOutput = check_output('./cms' + ' %s.cmcards' %(''.join(time.split(':'))), shell=True)

                print 'Simulation took %s ' % (DT.datetime.now() - dt)
                os.chdir(curdir)
                t = 1

            if analyzeFlag == True:
                print '**\nBegin Analyze Script %s ' % DT.datetime.now()
                if waveFlag:
                    CMSanalyze(time, inputDict=inputDict)
                if flowFlag:
                    CMSFanalyze(time, inputDict=inputDict)


            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/cmtb')
                    print 'moved %s ' % file

        except Exception, e:
            print '<< ERROR >> HAPPENED IN THIS TIME STEP '
            print e
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)
            os.chdir(curdir)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print '___________________\n________________\n___________________\n________________\n___________________\n________________\n'
    print 'USACE FRF Coastal Model Test Bed : CMS Wave and Flow'

    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.load(f)
    except:

        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/CMS_Input_example for example yaml file.')

    # add in defaults for inputDict
    if 'THREDDS' not in inputDict.keys():
        inputDict['THREDDS'] = 'FRF'
    if 'bathyLoc' not in inputDict.keys():
        inputDict['bathyLoc'] = 'integrated_bathy'
    if 'duration' not in inputDict.keys():
        inputDict['duration'] = 24
    if 'generateFlag' not in inputDict.keys():
        inputDict['generateFlag'] = True
    if 'pFlag' not in inputDict.keys():
        inputDict['pFlag'] = False
    if 'runFlag' not in inputDict.keys():
        inputDict['runFlag'] = True
    if 'analyzeFlag' not in inputDict.keys():
        inputDict['analyzeFlag'] = True
    if 'logfileLoc' not in inputDict.keys():
        inputDict['logfileLoc'] = inputDict['workingDirectory']
    if 'wave_time_step' not in inputDict.keys():
        inputDict['wave_time_step'] = 30
    if 'flow_time_step' not in inputDict.keys():
        inputDict['flow_time_step'] = 60
    if 'morph_time_step' not in inputDict.keys():
        inputDict['morph_time_step'] = inputDict['flow_time_step']

    Master_CMS_run(inputDict=inputDict)
