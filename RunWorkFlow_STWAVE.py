# -*- coding: utf-8 -*-
#!/home/number/anaconda2/bin/python
import matplotlib
matplotlib.use('Agg')
import datetime as DT
from subprocess import check_output
import numpy as np
from frontback.frontBackSTWAVE import STanalyze, STsimSetup
import os, getopt, sys, shutil, glob, platform, logging, yaml

def Master_STWAVE_run(inputDict):
    """This will run STWAVE with any version prefix given start, end, and timestep
    `   version_prefix, startTime, endTime, simulationDuration, all of which are in the input dictionary loaded from the
    input yaml file
    This is the generic work flow

    Args:
      inputDict:
        key pFlag: plots or not (boolean)
        :key analyzeFlag: analyze results or not (boolean)
        :key generateFlag: generate input files or not (boolean)
        :key runFlag: run the simulation or not (boolean)
        :key start_date: date I am starting the simulation (format '2018-01-15T00:00:00Z')
        :key end_date: date I am ending the simulation (format '2018-01-15T00:00:00Z')
        :key workingDirectory: path to the working directory the user wants
        :key netCDFdir: path to the netCDF save location specified by the user
        :key THREDDS: which THREDDS server are we using, 'FRF' or 'CHL'
        :key version_prefix: right now we have 'FIXED', 'MOBILE', or 'MOBILE_RESET'
        :key duration: how long you want the simulations to run in hours (24 by default)

    Returns:
      None

    """
    # globals:
    inputDict['model'] = inputDict['modelName']  # short cut
    ###################################################################################################################
    #######################   Parse out input Dictionary     ##########################################################
    ###################################################################################################################
    # required inputs
    generateFlag = inputDict['generateFlag']  # flag responsible for generating simulation files
    runFlag = inputDict['runFlag']  # flag responsible for running the simulation
    analyzeFlag = inputDict['analyzeFlag']  # flag responsible for rerunning analysis routine
    pFlag = inputDict['pFlag']  # flag for running  plots with the analysis routine
    version_prefix = inputDict['version_prefix']
    path_prefix = inputDict['workingDirectory']
    startTime = inputDict['startTime']
    endTime = inputDict['endTime']
    simulationDuration = inputDict['simulationDuration']

    if 'hostfileLoc' in inputDict:
        hostfile = inputDict['hostfileLoc']
    else:
        hostfile = 'hostfile-IB'

    ## handle Architecture here
    if 'ForcedSurveyDate' in list(inputDict.keys()):
        ForcedSurveyDate = inputDict['ForcedSurveyDate']
        path_prefix = os.path.join(inputDict['modelName'], version_prefix, 'SingleBathy_{}'.format(ForcedSurveyDate))
    else:
        ForcedSurveyDate = None
        path_prefix = os.path.join(inputDict['modelName'], version_prefix)

    ###################################################################################################################
    #######################   doing Data check and setting up input vars  #############################################
    ###################################################################################################################

    prefixList = np.array(['HP',        # half plane (operational)
                           'FP',        # full plan (operational)
                           'CB',        # cbathy Operational
                           'CBHP',      # Half plane run at 10 m (experimental
                           'CBThresh',  # RESERVED for operational Cbathy study results (expermiental)
                           'CBT2',      # Run cbathy with threshold, outside kalman filter (expermental)
                           'CBT1'])     # run cbathy with threshold, inside kalman filter ( experimental)
    assert (version_prefix == prefixList).any(), "Please enter a valid version prefix\n Prefix assigned = %s must be in List %s" % (version_prefix, prefixList)
    # __________________input vars________________________________
    executableLocation = inputDict['modelExecutable']
    if inputDict['workingDirectory'].endswith(inputDict['version_prefix']):
        simulation_workingDirectory = inputDict['workingDirectory']
    else:
        simulation_workingDirectory = os.path.join(inputDict['workingDirectory'],inputDict['version_prefix'] )
    if version_prefix == 'FP':
        nproc_par = 24
    else:
        nproc_par = 12
    nproc_nest = 4

    # auto generated Log file using start_end time
    LOG_FILENAME = simulation_workingDirectory+'logs/CMTB_BatchRun_Log_%s_%s_%s.log' %(version_prefix, startTime.replace(':',''), endTime.replace(':',''))
    # #
    # try:  # COMMENT THIS BLOCK to see error lines
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # except IOError:
    #     os.makedirs(simulation_workingDirectory+'logs')
    #     logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    # logging.debug('\n-------------------\nTraceback Error Log for:\n\nSimulation Started: %s\n-------------------\n'
    #               % (DT.datetime.now()))
    # ____________________________________________________________
    # establishing the resolution of the input datetime
    d2 = DT.datetime.strptime(endTime, '%Y-%m-%dT%H:%M:%SZ')
    d1 = DT.datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%SZ')
    ###################################################################################################################
    #######################   Creat list of input Dates to loop over for simulations ##################################
    ###################################################################################################################

    # This is the portion that creates a list of simulation end times
    dt_DT = DT.timedelta(0, simulationDuration * 60 * 60)  # timestep in datetime
    # make List of Datestring items, for simulations
    a = [d1]
    dateStringList = [a[0].strftime("%Y-%m-%dT%H:%M:%SZ")]
    for i in range(int(np.ceil((d2-d1).total_seconds()/dt_DT.total_seconds()))-1):
        a.append(a[-1] + dt_DT)
        dateStringList.append(a[-1].strftime("%Y-%m-%dT%H:%M:%SZ"))

    errors, errorDates = [],[]
    curdir = os.getcwd()
    # run the process through each of the above dates
    print('\n-\n-\nMASTER WorkFLOW for STWAVE SIMULATIONS\n-\n-\n')
    print('Batch Process Start: %s     Finish: %s '% (d1, d2))
    print('The batch simulation is Run in %s Version' % version_prefix)
    print('Check for simulation errors here %s' % LOG_FILENAME)
    print('------------------------------------\n\n************************************\n\n------------------------------------\n\n')
    ###################################################################################################################
    #######################   Loop over each day's simulation    ######################################################
    ###################################################################################################################
    for time in dateStringList:
        print(' ------------------------------ START %s --------------------------------' %time)

        try:
            datadir = os.path.join(simulation_workingDirectory, ''.join(time.split(':')))  # moving to the new simulation's folder
            if generateFlag == True:
                [nproc_par, nproc_nest] = STsimSetup(time, inputDict)

                if nproc_par == -1 or nproc_nest == -1:
                    print('************************\nNo Data available\naborting run\n***********************')
                    # remove generated files?
                    shutil.rmtree(simulation_workingDirectory+''.join(time.split(':')))
                    continue  # this is to return to the next time step if there's no cbathy data

            if runFlag == True:
                os.chdir(datadir)  # changing locations to where simulation files live
                t= DT.datetime.now()
                print('Beggining Parent Simulation %s' %t)
                try:
                    assert os.path.isfile(hostfile), 'Check hostfile path'
                    parent = check_output('mpiexec -n {} -f {} {} {}.sim'.format(nproc_par, hostfile, executableLocation, ''.join(time.split(':'))), shell=True)
                except AssertionError:
                    import multiprocessing
                    count = multiprocessing.cpu_count()  # Max out computer cores
                    parent = check_output('mpiexec -n {} {} {}.sim'.format(count, executableLocation, ''.join(time.split(':'))), shell=True)
                try:
                    assert os.path.isfile(hostfile), 'Check hostfile path'
                    child = check_output('mpiexec -n {} -f {} {} {}nested.sim'.format(nproc_nest, hostfile, executableLocation, ''.join(time.split(':'))), shell=True)
                except AssertionError:
                    import multiprocessing
                    count = multiprocessing.cpu_count()  #
                    child = check_output('mpiexec -n {} {} {}nested.sim'.format(count, executableLocation, ''.join(time.split(':'))), shell=True)
                    print(('  Simulations took {}'.format(DT.datetime.now() - t)))
            # run analyze and archive script
            os.chdir(curdir)  # change back after runing simulation locally
            if analyzeFlag == True:
                STanalyze(time, inputDict)
            if pFlag == True and DT.date.today() == d2.date():
                print('**\n Moving Plots! \n &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                # move files
                moveFnames = glob.glob(datadir + '/figures/CMTB*.png')
                moveFnames.extend(glob.glob(datadir + '/figures/CMTB*.gif'))
                for file in moveFnames:
                    shutil.copy(file,  '/mnt/gaia/gages/results/frfIn/CMTB')
                    print('moved %s ' % file)
            print(' --------------   SUCCESS: Done %s --------------------------------' %time)
        except Exception as e:
            os.chdir(curdir)  # if things break during run flag, need to get back out!
            print('<< ERROR >> HAPPENED IN THIS TIME STEP ')
            # print e
            print(e.args)
            logging.exception('\nERROR FOUND @ %s\n' %time, exc_info=True)

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    print('___________________\n________________\n___________________\n________________\n___________________\n________________\n')
    print('USACE FRF Coastal Model Test Bed : STWAVE')
    import yaml
    yamlLoc = args[0]
    with open(yamlLoc, 'r') as f:
        inputDict = yaml.load(f)  # load input yaml
    # run work flow
    Master_STWAVE_run(inputDict)
