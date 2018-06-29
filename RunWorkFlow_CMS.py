# -*- coding: utf-8 -*-
#!/home/number/anaconda2/bin/python
import matplotlib
matplotlib.use('Agg')
import os, getopt, sys, shutil, glob, logging, yaml
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackCMS import CMSanalyze
from frontback.frontBackCMS import CMSsimSetup


def Master_CMS_run(inputDict):
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

    # data check
    prefixList = np.array(['HP', 'UNTUNED'])
    assert (version_prefix == prefixList).any(), "Please enter a valid version prefix\n Prefix assigned = %s must be in List %s" % (version_prefix, prefixList)

    # __________________input directories________________________________
    codeDir = os.getcwd()  # location of code
    # check executable
    if inputDict['modelExecutable'].startswith(codeDir):  # change to relative path
        import re
        inputDict['modelExecutable'] = re.sub(codeDir, '', inputDict['modelExecutable'])

    if workingDir[-1] == '/':
        outDataBase = workingDir + 'CMS/' + version_prefix + '/'  #codeDir + '/%s_CSHORE_data/' % version_prefix
    else:
        outDataBase = workingDir + '/CMS/' + version_prefix +'/'

    inputDict['path_prefix'] = outDataBase
    TOD = 0  # 0=start simulations at 0000
    # ______________________ Logging  ____________________________
    # auto generated Log file using start_end time?

    LOG_FILENAME = outDataBase+'logs/cmtb_BatchRun_Log_%s_%s_%s.log' %(version_prefix, startTime, endTime)
    # try:
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # except IOError:
    #     os.makedirs(outDataBase+'logs')
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # logging.debug('\n-------------------\nTraceback Error Log for:\n\nSimulation Started: %s\n-------------------\n'
    #               % (DT.datetime.now()))
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    try:
        projectEnd = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
        projectStart = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ') + DT.timedelta(TOD / 24., 0, 0)
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
    print '\n-\n-\nMASTER WorkFLOW for STWAVE SIMULATIONS\n-\n-\n'
    print 'Batch Process Start: %s     Finish: %s '% (projectStart, projectEnd)
    print 'The batch simulation is Run in %s Version' % version_prefix
    print 'Check for simulation errors here %s' % LOG_FILENAME
    print '------------------------------------\n\n************************************\n\n------------------------------------\n\n'

    # ________________________________________________ RUN LOOP ________________________________________________
    for time in dateStringList:
        try:
            print '**\nBegin '
            print 'Beginning Simulation %s' %DT.datetime.now()

            if generateFlag == True:
                CMSsimSetup(time, inputDict=inputDict)
                datadir = outDataBase + ''.join(time.split(':'))  # moving to the new simulation's folder

            if runFlag == True: # run model
                os.chdir(datadir) # changing locations to where input files should be made
                print 'Running CMS Simulation'
                dt = DT.datetime.now()

                simOutput = check_output(codeDir + '%s %s.sim' %(inputDict['modelExecutable'], ''.join(time.split(':'))), shell=True)

                print 'Simulation took %s ' % (DT.datetime.now() - dt)
                os.chdir(curdir)

            if analyzeFlag == True:
                print '**\nBegin Analyze Script %s ' % DT.datetime.now()
                CMSanalyze(time, inputDict=inputDict)

            if pFlag == True and DT.date.today() == projectEnd:
                # move files
                moveFnames = glob.glob(curdir + 'cmtb*.png')
                moveFnames.extend(glob.glob(curdir + 'cmtb*.gif'))
                for file in moveFnames:
                    shutil.move(file,  '/mnt/gaia/cmtb')
                    print 'moved %s ' % file
            print('------------------SUCCESSS-----------------------------------------')

        except Exception, e:
            print '<< ERROR >> HAPPENED IN THIS TIME STEP '
            print e
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)
            os.chdir(curdir)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print '___________________\n________________\n___________________\n________________\n___________________\n________________\n'
    print 'USACE FRF Coastal Model Test Bed : CMS Wave'

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
