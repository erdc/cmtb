# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, re, pickle
import datetime as DT
import numpy as np
from frontback import frontBackNEW
from getdatatestbed.getDataFRF import getObs
from testbedutils import fileHandling
from prepdata import writeRunRead as wrrClass

def Master_ww3_run(inputDict):
    """This function will run CMS with any version prefix given start, end, and timestep.

    Args:
      inputDict: a dictionary that is read from the input yaml

    Returns:
      None

    """
    ## unpack input Dictionary
    version_prefix = inputDict['modelSettings']['version_prefix']
    endTime = inputDict['endTime']
    startTime = inputDict['startTime']
    simulationDuration = inputDict['simulationDuration']
    workingDir = os.path.join(inputDict['workingDirectory'], 'waveModels')
    generateFlag = inputDict['generateFlag']
    runFlag = inputDict['runFlag']
    analyzeFlag = inputDict['analyzeFlag']
    pFlag = inputDict['plotFlag']
    model = inputDict.get('model', 'ww3')
    log = inputDict.get('logging', True)

    # __________________pre-processing checks________________________________
    fileHandling.checkVersionPrefix(model, inputDict)
    # __________________input directories________________________________
    cmtbRootDir = os.getcwd()  # location of working directory
    # check executable
    if inputDict['modelExecutable'].startswith(cmtbRootDir):  # change to relative path
        inputDict['modelExecutable'] = re.sub(cmtbRootDir, '', inputDict['modelExecutable'])
    workingDirectory = os.path.join(workingDir, model.lower(), version_prefix)
    inputDict['netCDFdir'] = os.path.join(inputDict['netCDFdir'], 'waveModels')
    inputDict['path_prefix'] = workingDirectory
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end time?
    LOG_FILENAME = fileHandling.logFileLogic(workingDirectory, version_prefix, startTime, endTime, log=log)
    # __________________get time list to loop over________________________________
    dateStartList, dateStringList, projectStart, projectEnd = fileHandling.createTimeInfo(startTime, endTime,
                                                                                  simulationDuration=simulationDuration)
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, model)
    # ______________________________gather all data _____________________________
    if generateFlag == True:
        go = getObs(projectStart, projectEnd)  # initialize get observation
        rawspec = go.getWaveSpec(gaugenumber='waverider-26m', specOnly=True)
        rawWL = go.getWL()
        rawwind = go.getWind(gaugenumber=0)

    # ________________________________________________ RUN LOOP ________________________________________________
    # run the process through each of the above dates
    errors, errorDates, curdir = [], [], os.getcwd()
    for time in dateStringList:
        print('Beginning to setup simulation {}'.format(DT.datetime.now()))
        try:
            dateString = ''.join(time.split(':'))
            datadir = os.path.join(workingDirectory, dateString)  # moving to the new simulation's
            # pickleSaveName = os.path.join(datadir, timeStamp + '_ww3io.pickle')
            # # if generateFlag == True:
            #     ww3io = frontBackWW3.ww3simSetup(time, inputDict=inputDict, allWind=rawwind, allWL=rawWL,
            #                                      allWave=rawspec)
            #
            ####### THE NEW WAY!
            # load the instance of wrr # TBD later on what will control this
            # are there other things we need to load?
            wrr = wrrClass.ww3io(pathPrefix=datadir, fNameBase=dateString, versionPrefix=version_prefix,
                                 dateString=dateString, startTime=startTime, endTime=endTime, runFlag=runFlag,
                                 generateFlag=generateFlag, readFlag=analyzeFlag)
            
            wavePacket, windPacket, WLpacket, bathyPacket, gridFname, wrr = frontBackNEW.ww3simSetup(time,
                                                                                                inputDict=inputDict,
                                                                                                allWind=rawwind,
                                                                                                allWL=rawWL,
                                                                                                allWave=rawspec,
                                                                                                wrr=wrr)
            
            print('TODO: document Packets coming from sim-setup')
            # write simulation files (if assigned)
            wrr.writeAllFiles(wavePacket, windPacket, WLpacket, bathyPacket, gridFname)
            
            # run simulation (as appropriate)
            wrr.runSimulation(modelExecutable=inputDict['modelExecutable'])
            
            # post process (as appropriate)
            spatialData, savePointData = wrr.readAllFiles()

            if analyzeFlag == True:
                if generateFlag is False and runFlag is False:
                    try:  # to load the pickle
                        with open(wrr.pickleSaveName, 'rb') as fid:
                            ww3io = pickle.load(fid)
                    except(FileNotFoundError):
                        print("couldn't load sim metadata pickle for post-processing: moving to next time")
                        continue
                frontBackNEW.ww3analyze(time, inputDict=inputDict, ww3io=ww3io)

            # if it's a live run, move the plots to the output directory
            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                liveFileMoveToDirectory = '/mnt/gaia/cmtb'
                for file in moveFnames:
                    shutil.move(file,  liveFileMoveToDirectory)
                    print('moved {} to {} '.format(file, liveFileMoveToDirectory))
            print('------------------SUCCESS-----------------------------------------')

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ {}\n'.format(time, exc_info=True))
            os.chdir(curdir)


if __name__ == "__main__":
    model = 'ww3'
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________________________\n___________________________________\n___________________'
          '________________\n')
    print('USACE FRF Coastal Model Test Bed : {}'.format(model))

    try:
        # assume the user gave the path
        yamlLoc = args[0]
        if os.path.exists('.cmtbSettings'):
            with open('.cmtbSettings', 'r') as fid:
                a = yaml.safe_load(fid)
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.safe_load(f)
        inputDict.update(a)
        
    except:
        raise IOError('Input YAML file required. See yaml_files/TestBedExampleInputs/{}_Input_example for example yaml file.'.format(model))

    Master_ww3_run(inputDict=inputDict)
