# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackSWASH import SwashSimSetup
from frontback.frontBackSWASH import SwashAnalyze


def Master_SWASH_run(inputDict):
    """This function will run CMS with any version prefix given start, end, and timestep

    Args:
      inputDict: a dictionary that is read from the input yaml

    Returns:
      None

    """
    ## unpack Dictionary
    version_prefix = inputDict['version_prefix']
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['pFlag']
    model = inputDict['modelName']
    # data check
    prefixList = np.array(['base', 'ts'])
    assert (version_prefix.lower() == prefixList).any(), "Please enter a valid version prefix\n Prefix assigned = %s must be in List %s" % (version_prefix, prefixList)

    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
        import re
        inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])

    outDataBase = os.path.join(workingDir, model, version_prefix)
    inputDict['path_prefix'] = outDataBase
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end time
    LOG_FILENAME = os.path.join(outDataBase,'logs/{}_BatchRun_Log_{}_{}_{}.log'.format(model,version_prefix, startTime, endTime))
    # try:
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # except IOError:
    #     os.makedirs(os.path.join(outDataBase,'logs'))
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # logging.debug('\n-------------------\nTraceback Error Log for:\n\nSimulation Started: %s\n-------------------\n'
    #               % (DT.datetime.now()))
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    try:
        projectEnd = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ')
        projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        assert len(endTime) == 10, 'Your Time does not fit convention, check T/Z and input format'

    # This is the portion that creates a list of simulation end times
    dt_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    dateStartList = [projectStart]
    dateStringList = [dateStartList[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((projectEnd-projectStart).total_seconds()/dt_DT.total_seconds()))-1):
        dateStartList.append(dateStartList[-1] + dt_DT)
        dateStringList.append(dateStartList[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))

    errors, errorDates = [],[]
    curdir = os.getcwd()
    # ______________________________decide process and run _____________________________
    # run the process through each of the above dates
    print('\n-\n-\nMASTER WorkFLOW for {} SIMULATIONS\n-\n-\n'.format(model))
    print('Batch Process Start: %s     Finish: %s '% (projectStart, projectEnd))
    print('The batch simulation is Run in %s Version' % version_prefix)
    print('Check for simulation errors here {}'.format(LOG_FILENAME))
    print('------------------------------------\n\n************************************\n\n------------------------------------\n\n')

    # ________________________________________________ RUN LOOP ________________________________________________
    for time in dateStringList:
        try:
            print('**\nBegin ')
            print('Beginning Simulation %s' %DT.datetime.now())

            if generateFlag == True:
                SWIO = SwashSimSetup(time, inputDict=inputDict)
                datadir = os.path.join(outDataBase, ''.join(time.split(':')))  # moving to the new simulation's folder

            if runFlag == True:        # run model
                os.chdir(datadir)      # changing locations to where input files should be made
                print('Running Simulation')
                dt = DT.datetime.now()
                print(" use {} processors".format(SWIO.nprocess))
                simOutput = check_output("mpirun -n {} {}{} INPUT".format(SWIO.nprocess, codeDir, inputDict['modelExecutable']), shell=True)
                SWIO.simulationWallTime = DT.datetime.now() - dt
                print('Simulation took {}'.format(SWIO.simulationWallTime))
                os.chdir(curdir)

            if analyzeFlag == True:
                print('**\nBegin Analyze Script %s ' % DT.datetime.now())
                SwashAnalyze(time, inputDict, SWIO)

            if pFlag is True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/cmtb')
                    print('moved %s ' % file)
            print('------------------Model Run: SUCCESSS-----------------------------------------')

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)
            os.chdir(curdir)  # change back to main directory (no matter where the simulation failed)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed')

    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.load(f)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/CMS_Input_example for example yaml file.')

    Master_SWASH_run(inputDict=inputDict)
