# -*- coding: utf-8 -*-
#!/home/number/anaconda2/bin/python
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, logging
import numpy as np
from subprocess import check_output
import datetime as DT
from frontback.frontBackCSHORE import CSHORE_analysis, CSHOREsimSetup
from prepdata import inputOutput
from getdatatestbed.getDataFRF import getDataTestBed
import yaml
import platform
import netCDF4 as nc

def master_CSHORE_run(inputDict):
    """will run CSHORE with any version prefix given start, end, and a simulation duration

    Args:
      inputDict: keys are:
    :key pFlag - plots or not (boolean)
    :key analyzeFlag - analyze results or not (boolean)
    :key generateFlag - generate input files or not (boolean)
    :key runFlag - run the simulation or not (boolean)
    :key start_date - date I am starting the simulation (format '2018-01-15T00:00:00Z')
    :key end_date - date I am ending the simulation (format '2018-01-15T00:00:00Z')
    :key WD - path to the working directory the user wants
    :key netCDFdir - path to the netCDF save location specified by the user
    :key THREDDS - which THREDDS server are we using, 'FRF' or 'CHL'
    :key version_prefix - right now we have 'FIXED', 'MOBILE', or 'MOBILE_RESET'
    :key duration - how long you want the simulations to run in hours (24 by default)

    Returns:
      None

    """

    version_prefix = inputDict['version_prefix']
    endTime = inputDict['end_date']
    startTime = inputDict['start_date']
    simulationDuration = inputDict['simulationDuration']
    THREDDS = inputDict['THREDDS']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    sorceCodePATH = inputDict['modelExecutable']

    # version check
    prefixList = np.array(['FIXED', 'MOBILE', 'MOBILE_RESET'])
    assert (version_prefix == prefixList).any(), "Please enter a valid version prefix\n Prefix assigned = %s must be in List %s" % (version_prefix, prefixList)

    # __________________input vars________________________________
    codeDir = os.getcwd()
    if workingDir[-1] == '/':
        outDataBase =os.path.join(workingDir, 'CSHORE', version_prefix)
    else:
        outDataBase = os.path.join(workingDir, 'CSHORE', version_prefix)

    TOD = 0  # 0=start simulations at 0000
    LOG_FILENAME = os.path.join(inputDict['logfileLoc'], 'CSHORE/%s/logs/CMTB_BatchRun_Log_%s_%s_%s.log' %(version_prefix, version_prefix, startTime.replace(':',''), endTime.replace(':','')))
    #
    # try:
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # except IOError:
    #     try:
    #         os.makedirs(os.path.join(outDataBase,'logs')) # in python 3, exist_ok=True)
    #     except OSError:  #folder exists
    #         pass
    #     LOG_FILENAME = os.path.join(outDataBase, 'logs/CMTB_BatchRun_Log_%s_%s_%s.log' %(version_prefix, startTime.replace(':',''), endTime.replace(':','')))
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # logging.debug('\n-------------------\nTraceback Error Log for:\n\nSimulation Started: %s\n-------------------\n' % (DT.datetime.now()))
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    d2 = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)

    # if the version is MOBILE then I do NOT want to check this, because MOBILE continuously
    # evolves and NEVER resets the bathymetry
    if version_prefix != 'MOBILE':
        # pull the latest bathymetry
        cmtb_data = getDataTestBed(d1, d1+DT.timedelta(minutes=1), THREDDS)
        bathy_data = cmtb_data.getBathyIntegratedTransect()
        b_time = bathy_data['time']
        all_times = [nc.num2date(tt, 'seconds since 1970-01-01') for tt in cmtb_data.allEpoch]
        my_times = [tt for tt in all_times if tt >= d1 - DT.timedelta(hours=18)]
        my_times = [tt for tt in my_times if tt <= d2]
        t = 1

        try:         # try to pull the .nc file of the previous run.
            Time_O = (DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') - DT.timedelta(days=1)).strftime('%Y-%m-%dT%H%M%SZ')
            # initialize the class
            cshore_io_O = inputOutput.cshoreIO()
            # get into the directory I need
            params0, bc0, veg0, hydro0, sed0, morpho0, meta0 = cshore_io_O.load_CSHORE_results(outDataBase + Time_O)

            # what SURVEY TIME did this thing use??
            prev_mod_stime = meta0['bathy_surv_stime']

            # ok, this says that if:
            # 1 - the previous model used a survey older than the latest survey
            # 2 - the previous model started AFTER the latest survey (i.e., it should have used the latest survey)
            if (b_time > prev_mod_stime) and (Time_O > b_time):
                d1_N = b_time.replace(microsecond=0, second=0, minute=0, hour=0)
                if d1_N != b_time:
                    d1_N = d1_N + DT.timedelta(days=1)
                    # this means we rounded it down and have to add back a day to start on the 00:00:00 after the survey

                # reset the first day of the simulations to be the day after or of the latest survey
                # (depending on if the survey time is 00:00:00 or 12:00:00)
                del d1
                d1 = d1_N
                del d1_N

        except IOError:
            # this means that this is the first time this has been run, so you don't have to worry about it.
            pass


    # This is the portion that creates a list of simulation end times (start times?)
    dt_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    a = [d1]
    dateStringList = [a[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((d2-d1).total_seconds()/dt_DT.total_seconds()))-1):
        a.append(a[-1] + dt_DT)
        dateStringList.append(a[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))

    errors, errorDates = [],[]
    # change this to be the same as the data folder?
    # os.chdir(workingDir)  # is this right?  NO it's not
    curdir = os.getcwd()

    # figure out which days need to be reset days
    dateTimeList = [DT.datetime.strptime(tt, '%Y-%m-%dT%H:%M:%SZ') for tt in dateStringList]
    reset_list = np.zeros(np.shape(dateTimeList), dtype=bool)
    if version_prefix != 'MOBILE':
        for ss in range(0, len(dateTimeList)):
            check_date = [DT.timedelta(hours=12).total_seconds() >= abs(((dateTimeList[ss] - check_time) + DT.timedelta(minutes=1)).total_seconds()) for check_time in my_times]
            if any(check_date):
                reset_list[ss] = True


    cnt = 0
    for time in dateStringList:
        # tag if this is a reset day or not
        inputDict['reset'] = reset_list[cnt]
        cnt = cnt + 1
        try:
            print('----------------------Begin {} ---------------------------'.format(time))
            if generateFlag == True:
                CSHOREsimSetup(startTime=time, inputDict=inputDict)
                datadir = os.path.join(outDataBase, ''.join(time.split(':')))  # moving to the new simulation's folder

            if runFlag == True:
                os.chdir(datadir)# changing locations to where data should be downloaded to
                shutil.copy2(sorceCodePATH, datadir)
                print('Bathy Interpolation done\n Beginning Simulation')
                check_output(os.path.join('./', sorceCodePATH.split('/')[-1]), shell=True)
                # as this is written the script has to be in the working directory, not in a sub-folder!

            # run analyze and archive script
            os.chdir(curdir)
            if analyzeFlag == True:
                print('**\nBegin Analyze Script')
                CSHORE_analysis(startTime=time, inputDict=inputDict)

            # not sure i want this so i commented it out for now
            """
            if pFlag == True and DT.date.today() == d2:
                # move files
                moveFnames = glob.glob(curdir + 'CMTB*.png')
                moveFnames.extend(glob.glob(curdir + 'CMTB*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/CMTB')
                    print 'moved %s ' % file
            """
            print('----------------------SUCCESS--------------------')
        except Exception as e:
            os.chdir(curdir)
            print('   << ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : CSHORE')

    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.load(f)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/CSHORE_Input_example for example yaml file.')


    # add in defaults for inputDict
    if 'THREDDS' not in list(inputDict.keys()):
        inputDict['THREDDS'] = 'FRF'
    if 'bathyLoc' not in list(inputDict.keys()):
        inputDict['bathyLoc'] = 'integrated_bathy'
    if 'profileNumber' not in list(inputDict.keys()):
        inputDict['profileNumber'] = 960
    if 'duration' not in list(inputDict.keys()):
        inputDict['duration'] = 24
    if 'generateFlag' not in list(inputDict.keys()):
        inputDict['generateFlag'] = True
    if 'pFlag' not in list(inputDict.keys()):
        inputDict['pFlag'] = False
    if 'runFlag' not in list(inputDict.keys()):
        inputDict['runFlag'] = True
    if 'analyzeFlag' not in list(inputDict.keys()):
        inputDict['analyzeFlag'] = True
    if 'logfileLoc' not in list(inputDict.keys()):
        inputDict['logfileLoc'] = inputDict['workingDirectory']

    master_CSHORE_run(inputDict=inputDict)














