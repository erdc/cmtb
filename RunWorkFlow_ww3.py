# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml, re
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
    pFlag = inputDict['pFlag']
    model = inputDict.get('model', 'ww3')
    server = inputDict.get('THREDDS', 'CHL')

    # __________________pre-processing checks________________________________
    fileHandling.checkVersionPrefix(version_prefix, model)

    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
        inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])
    workingDirectory = os.path.join(workingDir, model.lower(),version_prefix)
    inputDict['path_prefix'] =  workingDirectory

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
    go = getObs(startTime, endTime, THREDDS=server)  # initialize get observation
    rawspec = go.getWaveSpec(gaugenumber=0)
    rawWL = go.getWL()
    rawwind = go.getWind(gaugenumber=0)

    # ________________________________________________ RUN LOOP ________________________________________________
    # run the process through each of the above dates
    errors, errorDates, curdir = [], [], os.getcwd()
    for time in dateStringList:
        try:
            print('Beginning Simulation %s' %DT.datetime.now())

            if generateFlag == True:
                gaugelocs = []
                # get gauge nodes x/y new idea: put gauges into input/output instance for the model, then we can save it
                for gauge in ww3io.waveGaugeList:
                    pos = go.getWaveGaugeLoc(gauge)
                    i, j = pos['lon'], pos['lat']
                    gaugelocs.append([i, j])

                ww3io = frontBackWW3.ww3simSetup(time, inputDict=inputDict, allWind=rawwind, allWL=rawWL, allWave=rawspec, gaugelocs=gaugelocs)
                datadir = workingDirectory + ''.join(time.split(':'))  # moving to the new simulation's folder

            if runFlag == True: # run model
                os.chdir(datadir) # changing locations to where input files should be made
                print('Running {} Simulation'.format(model))
                dt = DT.datetime.now()
                _ = check_output(codeDir + '%s %s.sim' %(inputDict['modelExecutable'], ''.join(time.split(':'))), shell=True)

                print('Simulation took %s ' % (DT.datetime.now() - dt))
                os.chdir(curdir)

            if analyzeFlag == True:
                print('**\nBegin Analyze Script %s ' % DT.datetime.now())
                frontBackWW3.ww3analyze(time, inputDict=inputDict)

            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/cmtb')
                    print('moved %s ' % file)
            print('------------------SUCCESSS-----------------------------------------')

        except Exception as e:
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            print(e)
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)
            os.chdir(curdir)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : CMS Wave')
    # we are no longer allowing a default yaml file.
    # It will throw and error and tell the user where to go look for the example yaml
    try:
        # assume the user gave the path
        yamlLoc = args[0]
        with open(os.path.join(yamlLoc), 'r') as f:
            inputDict = yaml.load(f)
    except:
        raise IOError('Input YAML file required.  See yaml_files/TestBedExampleInputs/{}_Input_example for example yaml file.'.format(model))

    Master_ww3_run(inputDict=inputDict)
