# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, time, pickle
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackSWASH import SwashSimSetup
from frontback.frontBackSWASH import SwashAnalyze
from testbedutils import fileHandling

def Master_SWASH_run(inputDict):
    """This function will run CMS with any version prefix given start, end, and timestep

    Args:
      inputDict: a dictionary that is read from the input yaml

    Returns:
      None

    """
    ## unpack Dictionary
    version_prefix = inputDict['version_prefix'].lower()
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = inputDict['workingDirectory']
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['pFlag']
    model = inputDict.get('modelName', 'SWASH').lower()
    inputDict['path_prefix'] = os.path.join(workingDir, model, version_prefix)
    path_prefix = inputDict['path_prefix']
    # data check
    prefixList = np.array(['base', 'ts'])
    assert (version_prefix.lower() == prefixList).any(), "Please enter a valid version prefix\n Prefix assigned = %s must be in List %s" % (version_prefix, prefixList)

    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    # if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
    #     import re
    #     inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])

    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end timeSegment
    LOG_FILENAME = os.path.join(path_prefix,'logs/{}_BatchRun_Log_{}_{}_{}.log'.format(model,version_prefix, startTime, endTime))
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
    for timeSegment in dateStringList:
        fileHandling.makeCMTBfileStructure(path_prefix=path_prefix, date_str=timeSegment)
        try:
            timeStamp = ''.join(timeSegment.split(':'))
            datadir = os.path.join(path_prefix, timeStamp)  # moving to the new simulation's folder
            pickleSaveFname = os.path.join(datadir, timeStamp + '_io.pickle')
            if generateFlag == True:
                SWIO = SwashSimSetup(timeSegment, inputDict=inputDict)

            if runFlag == True:        # run model
                os.chdir(datadir)      # changing locations to where input files should be made
                dt = time.time()
                print('Running Simulation started with {} processors'.format(SWIO.nprocess))
                _ = check_output("mpirun -n {} {} INPUT".format(SWIO.nprocess, os.path.join(codeDir, inputDict['modelExecutable'])), shell=True)
                SWIO.simulationWallTime = time.time() - dt
                print('Simulation took {:.1} seconds'.format(SWIO.simulationWallTime))
                os.chdir(curdir)
                with open(pickleSaveFname, 'wb') as fid:
                    pickle.dump(SWIO, fid, protocol=pickle.HIGHEST_PROTOCOL)

            else:   # assume there is a saved pickle of input/output that was generated before
                with open(pickleSaveFname, 'rb') as fid:
                    SWIO = pickle.load(fid)

            if analyzeFlag == True:
                print('**\nBegin Analyze Script %s ' % DT.datetime.now())
                SWIO.path_prefix = os.path.join(workingDir, model, version_prefix, timeStamp)
                SwashAnalyze(timeSegment, inputDict, SWIO)

            if pFlag is True and DT.date.today() == projectEnd:
                print('  TODO tar simulation files after generating netCDF')
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
            logging.exception('\nERROR FOUND @ %s\n' %timeSegment, exc_info=True)
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
            inputDict = yaml.safe_load(f)
        if os.path.exists('.cmtbSettings'):
            with open('.cmtbSettings', 'r') as fid:
                a = yaml.safe_load(fid)
            inputDict.update(a)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/CMS_Input_example for example yaml file.')

    Master_SWASH_run(inputDict=inputDict)
