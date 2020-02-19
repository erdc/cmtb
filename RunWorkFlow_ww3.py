# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, re, pickle
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback import frontBackWW3
from getdatatestbed.getDataFRF import getObs
from testbedutils import fileHandling

def Master_ww3_run(inputDict):
    """This function will run CMS with any version prefix given start, end, and timestep

    Args:
      inputDict: a dictionary that is read from the input yaml

    Returns:
      None

    """
    ## unpack input Dictionary
    version_prefix = inputDict['version_prefix']
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['plotFlag']
    model = inputDict.get('model', 'ww3')
    server = inputDict.get('THREDDS', 'CHL')

    # __________________pre-processing checks________________________________
    fileHandling.checkVersionPrefix(model, version_prefix)
    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
        inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])
    workingDirectory = os.path.join(workingDir, model.lower(), version_prefix)
    inputDict['path_prefix'] = workingDirectory
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end time?
    LOG_FILENAME = fileHandling.logFileLogic(workingDirectory, version_prefix, startTime, endTime, log=False)
    # __________________get time list to loop over________________________________
    projectEnd = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ')
    projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    # This is the portion that creates a list of simulation end times
    dt_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    dateStartList = [projectStart]
    dateStringList = [dateStartList[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((projectEnd-projectStart).total_seconds()/dt_DT.total_seconds()))-1):
        dateStartList.append(dateStartList[-1] + dt_DT)
        dateStringList.append(dateStartList[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, model)

    # ______________________________gather all data _____________________________
    go = getObs(projectStart, projectEnd, THREDDS=server)  # initialize get observation
    rawspec = go.getWaveSpec(gaugenumber='waverider-26m')
    rawWL = go.getWL()
    rawwind = go.getWind(gaugenumber=0)

    # ________________________________________________ RUN LOOP ________________________________________________
    # run the process through each of the above dates
    errors, errorDates, curdir = [], [], os.getcwd()
    for time in dateStringList:
        try:
            print('Beginning Simulation {}'.format(DT.datetime.now()))

            if generateFlag == True:
                ww3io = frontBackWW3.ww3simSetup(time, inputDict=inputDict,
                                                 allWind=rawwind, allWL=rawWL, allWave=rawspec)
                datadir = os.path.join(workingDirectory, ''.join(time.split(':')))  # moving to the new simulation's
                                                                                     # folder
                pickleSaveName = os.path.join(ww3io.path_prefix + '_ww3io.pickle')

            if runFlag == True:    # run model
                os.chdir(datadir)  # changing locations to where input files should be made
                dt = DT.datetime.now()
                print('Running {} Simulation starting at {}'.format(model, dt))
                runString = codeDir + '{} {}.sim'.format(inputDict['modelExecutable'], ''.join(time.split(':')))
                _ = check_output(runString, shell=True)
                ww3io.simulationWallTime = DT.datetime.now() - dt
                print('Simulation took {:.1f} minutes'.format(ww3io.simulationWallTime.total_seconds()/60))
                os.chdir(curdir)
                with open(pickleSaveName, 'wb') as fid:
                    pickle.dump(ww3io, fid, protocol=pickle.HIGHEST_PROTOCOL)

            if analyzeFlag == True:
                if generateFlag is False and runFlag is False:
                    try:  # to load the pickle
                        with open(pickleSaveName, 'rb') as fid:
                            ww3io = pickle.load(fid)
                    except (FileNotFoundError):
                        print("couldn't load sim metadata pickle for post-processing: moving to next time")
                        continue
                frontBackWW3.ww3analyze(time, inputDict=inputDict, ww3io=ww3io)

            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                liveFileMoveToDirectory = '/mnt/gaia/cmtb'
                for file in moveFnames:
                    shutil.move(file,  liveFileMoveToDirectory)
                    print('moved {} to {} '.format(file, liveFileMoveToDirectory))
            print('------------------SUCCESSS-----------------------------------------')

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ {}\n'.format(time, exc_info=True))
            os.chdir(curdir)


if __name__ == "__main__":
    model = 'ww3'
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : {}'.format(model))
    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.safe_load(f)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/{}_Input_example for example yaml file.'.format(model))

    Master_ww3_run(inputDict=inputDict)
