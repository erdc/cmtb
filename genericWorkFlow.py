# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, re, pickle
import datetime as DT
import numpy as np
from frontback import frontBackNEW
from getdatatestbed.getDataFRF import getObs
from testbedutils import fileHandling
from prepdata import writeRunRead as wrrClass

def Master_workFlow(inputDict):
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
    modelName = inputDict['modelSettings'].get('modelName', None)
    log = inputDict.get('logging', True)

    # __________________pre-processing checks________________________________
    fileHandling.checkVersionPrefix(modelName, inputDict)
    # __________________input directories________________________________
    cmtbRootDir = os.getcwd()  # location of working directory
    workingDirectory = os.path.join(workingDir, modelName.lower(), version_prefix)
    inputDict['netCDFdir'] = os.path.join(inputDict['netCDFdir'], 'waveModels')
    inputDict['path_prefix'] = workingDirectory
    # ______________________ Logging/FileHandling ____________________________
    # auto generated Log file using start_end time?
    LOG_FILENAME = fileHandling.logFileLogic(workingDirectory, version_prefix, startTime, endTime, log=log)
    dateStartList, dateStringList, projectStart, projectEnd = fileHandling.createTimeInfo(startTime, endTime,
                                                                                  simulationDuration=simulationDuration)
    fileHandling.displayStartInfo(projectStart, projectEnd, version_prefix, LOG_FILENAME, modelName)
    
    # ______________________________gather all data _____________________________
    if generateFlag == True:
        go = getObs(projectStart, projectEnd) # initialize get observation
        if modelName in ['ww3']:
            gauge = 'waverider-26m'
        elif modelName.lower() in ['swash', 'funwave']:
            gauge = '8m-array'
        elif modelName.lower() in ['cshore']:
            gauge = 'awac-6m'
            
        rawspec = go.getWaveData(gaugenumber=gauge, spec=True)
        rawWL = go.getWL()
        rawwind = go.getWind(gaugenumber=0)

    # ________________________________________________ RUN LOOP ________________________________________________
    # run the process through each of the above dates
    errors, errorDates = [], []
    for time in dateStringList:
        print('Beginning to setup simulation {}'.format(DT.datetime.now()))
        try:
            dateString = ''.join(''.join(time.split(':')).split('-'))
            # datadir = os.path.join(workingDirectory, dateString)  # moving to the new simulation's
            # pickleSaveName = os.path.join(datadir, timeStamp + '_ww3io.pickle')
            # # if generateFlag == True:
            #     ww3io = frontBackWW3.ww3simSetup(time, inputDict=inputDict, allWind=rawwind, allWL=rawWL,
            #                                      allWave=rawspec)
            #
            ####### THE NEW WAY!
            # load the instance of wrr # TBD later on what will control this
            # are there other things we need to load?

            print("TODO: Ty here you are creating a function that initalizes wrr and preps irregardless of model")
            if modelName in ['ww3']:
                wrr = wrrClass.ww3io(fNameBase=dateString, versionPrefix=version_prefix,
                                     startTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ'),
                                     endTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(
                                     hours=inputDict['simulationDuration']), runFlag=runFlag,
                                     generateFlag=generateFlag, readFlag=analyzeFlag)
                if generateFlag is True:
                    wavePacket, windPacket, WLpacket, bathyPacket, gridFname, wrr = frontBackNEW.ww3simSetup(time,
                                                                                                     inputDict=inputDict,
                                                                                                     allWind=rawwind,
                                                                                                     allWL=rawWL,
                                                                                                     allWave=rawspec,
                                                                                                     wrr=wrr)
                
            elif modelName in ['swash']:
                wrr = wrrClass.swashIO(fNameBase=dateString, versionPrefix=version_prefix,
                                       startTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ'),
                                       simulatedRunTime=inputDict['simulationDuration'],
                                       endTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(
                                           hours=inputDict['simulationDuration']), runFlag=runFlag,
                                       generateFlag=generateFlag, readFlag=analyzeFlag)
                if generateFlag is True:

                    wavePacket, windPacket, WLpacket, bathyPacket, gridFname, wrr = frontBackNEW.swashSimSetup(time,
                                                                                                     inputDict=inputDict,
                                                                                                     allWind=rawwind,
                                                                                                     allWL=rawWL,
                                                                                                     allWave=rawspec,
                                                                                                     wrr=wrr)
            elif modelName in ['cshore']:
                import pdb
                #pdb.set_trace()
                wrr = wrrClass.cshoreio(fNameBase=dateString, versionPrefix=version_prefix,
                                       startTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ'),
                                       simulatedRunTime=inputDict['simulationDuration'],
                                       endTime=DT.datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(
                                           hours=inputDict['simulationDuration']), runFlag=runFlag,
                                       generateFlag=generateFlag, readFlag=analyzeFlag)
                if generateFlag is True:
                    rawbathy = go.getBathyTransectFromNC(profilenumbers=960)
                    rawctd = go.getCTD()
                    wavePacket, windPacket, wlPacket, bathyPacket, ctdPacket, wrr = frontBackNEW.cshoreSimSetup(time,
                                                                                                     inputDict=inputDict,
                                                                                                     allWind=rawwind,
                                                                                                     allWL=rawWL,
                                                                                                     allWave=rawspec,
                                                                                                     allBathy=rawbathy,
                                                                                                     allCTD=rawctd,
                                                                                                     wrr=wrr)
            if generateFlag is True:
                print(" TODO: TY you're handing me back the same prepdata packets from all frontBacks")
                print('TODO: document Packets coming from sim-setup')
                try:
                    print('    PrepData Dicts below')
                    print("wavePacket has keys: {}".format(wavePacket.keys()))
                    print("WLPacket has keys: {}".format(WLpacket.keys()))
                    print("bathyPacket has keys: {}".format(bathyPacket.keys()))
                    print("windPacket has keys: {}".format(windPacket.keys()))
                except AttributeError:
                    pass
                  
                # write simulation files (if assigned)
                wrr.writeAllFiles(bathyPacket, wavePacket, wlPacket=wlPacket, ctdPacket=ctdPacket)
                
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
                frontBackNEW.genericPostProcess(time, inputDict=inputDict, ww3io=ww3io)

            # if it's a live run, move the plots to the output directory
            if pFlag is True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(cmtbRootDir + 'cmtb*.png')
                moveFnames.extend(glob.glob(cmtbRootDir + 'cmtb*.gif'))
                liveFileMoveToDirectory = '/mnt/gaia/cmtb'
                for file in moveFnames:
                    shutil.move(file,  liveFileMoveToDirectory)
                    print('moved {} to {} '.format(file, liveFileMoveToDirectory))
            print('------------------SUCCESS-----------------------------------------')

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ {}\n'.format(time, exc_info=True))
            os.chdir(cmtbRootDir)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________________________\n___________________________________\n___________________'
          '________________\n')
    print('USACE FRF Coastal Model Test Bed :')

    try:
        # assume the user gave the path
        yamlLoc = args[0]
        if os.path.exists('.cmtbSettings'):
            with open('.cmtbSettings', 'r') as fid:
                a = yaml.safe_load(fid)
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.safe_load(f)
        inputDict.update(a)
        #TODO: re-examine if input yaml properly overwrites .cmtbsettings values
        
    except:
        raise IOError('Input YAML file required. See yaml_files/TestBedExampleInputs/{}_Input_example for example yaml file.'.format(model))

    Master_workFlow(inputDict=inputDict)
