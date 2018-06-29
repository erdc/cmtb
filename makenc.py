"""
Created on 2/19/2016
This script is desinged to create netCDF files using the netCDF4 module from python as
part of the Coastal Model Test Bed (CMTB)

@author: Spicer Bak
@contact: Spicer.Bak@usace.army.mil
"""
import numpy as np
import netCDF4 as nc
import csv, yaml
import datetime as DT
import time as ttime

def readflags(flagfname, header=1):
    """This function reads the flag file from the data in to the STWAVE CMTB runs

    Args:
      flagfname: the relative/absolute location of the flags file
      header:  (Default value = 1)

    Returns:
      flags of data dtype=dictionary

    """
    times, waveflag, windflag, WLflag, curflag,allflags = [], [],[],[],[],[]

    try:
        with open(flagfname, 'rb') as f:
            reader = csv.reader(f)  # opening file
            for row in reader:  # iteratin
                # go over the open file
                if len(row) > 1 and row[0] != 'Date':
                    waveflag.append(int(row[2]))  # appending wave data flag
                    windflag.append(int(row[3]))  # appending Wind data flag
                    WLflag.append(int(row[4]))  # appending Water Level Flag data
                    curflag.append(int(row[5]))  # appending ocean Currents flag data
                    times.append(DT.datetime.strptime(row[0]+row[1], '%Y-%m-%d%H%M'))
                    allflags.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])])
        # creating array of flags
        allflags = np.array(allflags)
    except IOError:
        allflags = None
    # putting data into a dictionary
    flags = {'time': times,
             'windflag': windflag,
             'waveflag': waveflag,
             'WLflag': WLflag,
             'curflag': curflag,
             'allflags': allflags
             }
    return flags

def import_template_file(yaml_location):
    """This function loads a yaml file and returns the attributes in dictionary
    written by: ASA


    Step 1 in netCDF file creation, open global and variable yamls

    Args:
      yaml_location: yaml file location

    Returns:
        dictionary with variables from the yaml file

    """
    # load the template
    f = open(yaml_location)
    # use safe_load instead load
    vars_dict = yaml.safe_load(f)
    f.close()
    return vars_dict

def init_nc_file(nc_filename, attributes):
    """Create the netCDF file and write the Global Attributes
    written by ASA

    will initalize netCDF file and set global attributes, write date created and issued to global meta data

    Args:
      nc_filename: output netCDF file name
      attributes: attributes from global yaml load

    Returns:
        open netCDF file ready for writing data

    """

    ncfile = nc.Dataset(nc_filename, 'w', clobber=True)

    # Write some Global Attributes
    for key, value in attributes.iteritems():

        if value is not None:
            setattr(ncfile, key, value)

    dt_today = ttime.strftime("%Y-%m-%d")
    ncfile.date_created = dt_today
    ncfile.date_issued = dt_today

    return ncfile

def write_data_to_nc(ncfile, template_vars, data_dict, write_vars='_variables'):
    """This function actually writes the variables and the variable attributes to
    the netCDF file

    in the yaml, the "[variable]:" needs to be in the data dictionary,
     the output netcdf variable will take the name "name:"
    
     Edited by Spicer Bak

    Args:
      ncfile: this is an alreayd opened netCDF file with already defined dimensions
      template_vars (dict): variable and meta data associated with data_dict
      data_dict (dict): this is a dictionary with keys associated to those hopefully in template_vars, this holds the data
      write_vars: Unknown (Default value = '_variables')

    Returns:
      netCDF file (still open)
      also returns error strings and count that were created during the data writing process

    """
    # Keep track of any errors found
    num_errors = 0
    error_str = ''

    # write some more global attributes if present
    if '_attributes' in template_vars:
        for var in template_vars['_attributes']:
            if var in data_dict:
                setattr(ncfile, var, data_dict[var])

    # List all possible variable attributes in the template
    possible_var_attr = ['standard_name', 'long_name', 'coordinates', 'flag_values', 'flag_meanings', 'description',
                         'notes', 'positive', 'valid_min', 'valid_max', 'calendar', 'description', 'cf_role',
                         'missing_value']

    # Write variables to file
    accept_vars = template_vars['_variables']

    for var in accept_vars:  # only write varibles that were loaded from .yaml file
        if var in data_dict:
            try:
                if "fill_value" in template_vars[var] and "least_significant_digit" in template_vars[var]:
                    new_var = ncfile.createVariable(template_vars[var]["name"],
                                                    template_vars[var]["data_type"],
                                                    template_vars[var]["dim"],
                                                    fill_value=template_vars[var]["fill_value"],
                                                    least_significant_digit=template_vars[var]['least_significant_digit'] )
                elif "fill_value" in template_vars[var]:
                    new_var = ncfile.createVariable(template_vars[var]["name"], template_vars[var]["data_type"],
                            template_vars[var]["dim"], fill_value=template_vars[var]["fill_value"])
                else:
                    new_var = ncfile.createVariable(template_vars[var]["name"],
                                                    template_vars[var]["data_type"],
                                                    template_vars[var]["dim"])

                new_var.units = template_vars[var]["units"]

                # Write the attributes
                for attr in possible_var_attr:  # only write attributes listed in this list above
                    if attr in template_vars[var]:
                        if template_vars[var][attr] == 'NaN':
                            setattr(new_var, attr, np.nan)
                        else:
                            setattr(new_var, attr, template_vars[var][attr])
                # Write the short_name attribute as the variable name
                if 'short_name' in template_vars[var]:
                    new_var.short_name = template_vars[var]["short_name"]
                else:
                    new_var.short_name = template_vars[var]["name"]
                # _____________________________________________________________________________________
                # Write the data (1D, 2D, or 3D)
                #______________________________________________________________________________________
                if var == "station_name":
                    station_id = data_dict[var]
                    data = np.empty((1,), 'S'+repr(len(station_id)))
                    data[0] = station_id
                    new_var[:] = nc.stringtochar(data)
                elif len(template_vars[var]["dim"]) == 0:
                    try:
                        new_var[:] = data_dict[var]
                    except Exception, e:
                        new_var = data_dict[var]

                elif len(template_vars[var]["dim"]) == 1:
                    # catch some possible errors for frequency and direction arrays
                    if template_vars[var]["data_type"] == 'str':
                        for i, c in enumerate(template_vars[var]["data_type"]):
                            new_var[i] = data_dict[var][i]
                    else:
                        try:
                            new_var[:] = data_dict[var]
                        except IndexError:
                            try:
                                new_var[:] = data_dict[var][0][0]
                            except Exception, e:
                                raise e

                elif len(template_vars[var]["dim"]) == 2:
                    # create an empty 2d data set of the correct sizes
                    try:
                        # handles row vs col data, rather than transposing the array just figure out which it is
                        length = data_dict[var][0].shape[1]
                        if data_dict[var][0].shape[0] > length:
                            length = data_dict[var][0].shape[0]

                        x = np.empty([data_dict[var].shape[0], length], dtype=np.float64)
                        for i in range(data_dict[var].shape[0]):
                            # squeeze the 3d array in to 2d as dimension is not needed
                            x[i] = np.squeeze(data_dict[var][i])
                        new_var[:, :] = x
                    except Exception, e:
                        # if the tuple fails must be right...right?
                        new_var[:] = data_dict[var]

                elif len(template_vars[var]["dim"]) == 3:
                    # create an empty 3d data set of the correct sizes
                    # this portion was modified by Spicer Bak
                    assert data_dict[var].shape == new_var.shape, 'The data must have the Same Dimensions  (missing time?)'
                    x = np.empty([data_dict[var].shape[0], data_dict[var].shape[1], data_dict[var].shape[2]], np.float64)
                    for i in range(data_dict[var].shape[0]):
                        x[i] = data_dict[var][i]
                    new_var[:, :, :] = x[:, :, :]

            except Exception, e:
                num_errors += 1
                print('ERROR WRITING VARIABLE: {} - {} \n'.format(var, str(e)))


    return num_errors, error_str

def makenc_field(data_lib, globalyaml_fname, flagfname, ofname, var_yaml_fname):
    """This is a function that takes wave nest dictionary and Tp_nest dictionnary and creates the high resolution
    near shore field data from the Coastal Model Test Bed

    Args:
      data_lib: data lib is a library of data with keys the same name as associated variables to be written in the
            netCDF file to be created, This function will look for:

            'time', 'DX', 'DY', 'NI', 'NJ', 'bathymetry', 'bathymetryDate', 'waveHs', 'station_name'

      globalyaml_fname: global meta data yaml file name
      ofname: the file name to be created
      flagfname: flag input file to flag data
      var_yaml_fname:  variable meta data yaml file name


    Returns:
        written netCDF file
    """

    # import global atts
    globalatts = import_template_file(globalyaml_fname)
    # import variable data and meta
    var_atts = import_template_file(var_yaml_fname)
    # import flag data
    flags = readflags(flagfname)['allflags']
    data_lib['flags'] = flags
    # figure out my grid spacing and write it to the file
    if np.mean(data_lib['DX']) != np.median(data_lib['DX']):  # variable grid spacing
        globalatts['grid_dx'] = 'variable'
        globalatts['grid_dy'] = 'variable'
    else:
        globalatts['grid_dx'] = data_lib['DX']
        globalatts['grid_dy'] = data_lib['DY']
    globalatts['n_cell_y'] = data_lib['NJ']
    globalatts['n_cell_x'] = data_lib['NI']

    # making bathymetry the length of time so it can be concatnated
    if data_lib['waveHs'].shape[1] != data_lib['bathymetry'].shape[1]:
        data_lib['waveHs']=data_lib['waveHs'][:,:data_lib['bathymetry'].shape[1],:]
    data_lib['bathymetry'] = np.full_like(data_lib['waveHs'], data_lib['bathymetry'], dtype=np.float32 )
    if 'bathymetryDate' in data_lib:
        data_lib['bathymetryDate'] = np.full_like(data_lib['time'], data_lib['bathymetryDate'], dtype=np.float32 )


    #data_lib['bathymetry'] =
    fid = init_nc_file(ofname, globalatts)  # initialize and write inital globals

    #### create dimensions
    tdim = fid.createDimension('time', np.shape(data_lib['waveHs'])[0])
    xdim = fid.createDimension('X_shore', data_lib['NI'])
    ydim = fid.createDimension('Y_shore', data_lib['NJ'])
    inputtypes = fid.createDimension('in_type', np.shape(flags)[1]) # there are 4 input data types for flags
    statnamelen = fid.createDimension('station_name_length', len(data_lib['station_name']))
    #if 'bathymetryDate' in data_lib:
    #    bathyDate_length = fid.createDimension('bathyDate_length', np.shape(data_lib['bathymetry'])[0])

    # bathydate = fid.createDimension('bathyDate_length', np.size(data_lib['bathymetryDate']))

    # write data to the nc file
    write_data_to_nc(fid, var_atts, data_lib)
    # close file
    fid.close()

def makenc_FRFTransect(bathyDict, ofname, globalYaml, varYaml):
    """This function makes netCDF files from csv Transect data library created with testbedUtils.load_FRF_transect

    Args:
      bathyDict: data input matching var yaml, must have 'time' in it for dimension
      ofname: the file name to be created
      globalYaml: global meta data yaml file name
      varYaml: variable meta data yaml file name

    Returns:
        closed netCDF file

    """
    globalAtts = import_template_file(globalYaml)  # loading global meta data attributes from  yaml
    varAtts = import_template_file(varYaml)  # loading variables to write and associated meta data

    # initializing output cshore_ncfile
    fid =init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    tdim = fid.createDimension('time', np.shape(bathyDict['time'])[0])

    # write data to the cshore_ncfile
    write_data_to_nc(fid, varAtts, bathyDict)
    # close file
    fid.close()

def makenc_FRFGrid(gridDict, ofname, globalYaml, varYaml):
    """This is a function that makes netCDF files from the FRF Natural neighbor tool created by
    Spicer Bak using the pyngl library. the transect dictionary is created using the natural
    neighbor tool in FRF_natneighbor.py

    Args:
      gridDict: data dictionary matching varYaml requires
        'zgrid', 'ygrid', 'xgrid', 'StateplaneE', 'StateplaneN', 'Lat', 'Lon', 'FRF_X', 'FRF_Y'
      globalYaml: global meta data yaml file name
      ofname: the file name to be created
      varYaml: variable meta data yaml file name

    Returns:
      netCDF file with gridded data in it

    """
    from testbedutils import geoprocess as gp  # this might be creating a circular import
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xShore = fid.createDimension('xShore', np.shape(gridDict['zgrid'])[0])
    yShore = fid.createDimension('yShore', np.shape(gridDict['zgrid'])[1])
    time = fid.createDimension('time', np.size(gridDict['time']))

    # creating lat/lon and state plane coords
    #xgrid, ygrid = np.meshgrid(gridDict['xgrid'], gridDict['ygrid'])
    xx, yy = np.meshgrid(gridDict['xgrid'], gridDict['ygrid'])
    latGrid = np.zeros(np.shape(yy))
    lonGrid = np.zeros(np.shape(xx))
    statePlN = np.zeros(np.shape(yy))
    statePlE = np.zeros(np.shape(xx))
    for iy in range(0, np.size(gridDict['zgrid'], axis=1)):
        for ix in range(0, np.size(gridDict['zgrid'], axis=0)):
            coords = gp.FRFcoord(xx[iy, ix], yy[iy, ix])#, grid[iy, ix]))
            statePlE[iy, ix] = coords['StateplaneE']
            statePlN[iy, ix] = coords['StateplaneN']
            latGrid[iy, ix] = coords['Lat']
            lonGrid[iy, ix] = coords['Lon']
            assert xx[iy, ix] == coords['FRF_X']
            assert yy[iy, ix] == coords['FRF_Y']

    # put these data into the dictionary that matches the yaml
    gridDict['Latitude'] = latGrid[:, 0]
    gridDict['Longitude'] = lonGrid[0, :]
    gridDict['Easting'] = statePlE[:, 0]
    gridDict['Northing'] = statePlN[0, :]
    gridDict['FRF_Xshore'] = gridDict.pop('xgrid')
    gridDict['FRF_Yshore'] = gridDict.pop('ygrid')
    # addding 3rd dimension for time
    a=gridDict.pop('zgrid').T
    gridDict['Elevation'] = np.full([1, a.shape[0], a.shape[1]], fill_value=[a], dtype=np.float32)
    # write data to file
    write_data_to_nc(fid, varAtts, gridDict)
    # close file
    fid.close()

def makenc_Station(stat_data, globalyaml_fname, flagfname, ofname, stat_yaml_fname):
    """This function will make netCDF files from the station output data from the
    Coastal Model Test Bed of STWAVE for the STATion files

    Args:
      stat_data: data lib is a library of data with keys the same name as associated variables to be written in the
            netCDF file to be created, This function will look for:

            'time', 'DX', 'DY', 'NI', 'NJ', 'station_name', 'Northing', 'Easting', 'Longitude', 'Latitude', 'waveDirectionBins', 'waveFrequency'
      flagfname: name/path of flag file
      globalyaml_fname: global yaml name
      stat_yaml_fname: varable yamle name
      ofname: output file name

    Returns:
      a nc file with station data in it

    """
     # import global yaml data
    globalatts = import_template_file(globalyaml_fname)
    # import variable data and meta
    stat_var_atts = import_template_file(stat_yaml_fname)
    # import flag data
    flags = readflags(flagfname)['allflags']
    stat_data['flags'] = flags # this is a library of flags
    globalatts['grid_dx'] = stat_data['DX']
    globalatts['grid_dy'] = stat_data['DY']
    globalatts['n_cell_y'] = stat_data['NJ']
    globalatts['n_cell_x'] = stat_data['NI']
    fid = init_nc_file(ofname, globalatts)  # initialize and write inital globals

    #### create dimensions
    tdim = fid.createDimension('time', np.shape(stat_data['time'])[0])  # None = size of the dimension, what does this gain me if i know it
    inputtypes = fid.createDimension('input_types_length', np.shape(flags)[1]) # there are 4 input dtaa types for flags
    statnamelen = fid.createDimension('station_name_length', len(stat_data['station_name']))
    northing = fid.createDimension('Northing', 1L)
    easting = fid.createDimension('Easting', 1L )
    Lon = fid.createDimension('Longitude', np.size(stat_data['Longitude']))
    Lat = fid.createDimension('Latitude', np.size(stat_data['Latitude']))
    dirbin = fid.createDimension('waveDirectionBins', np.size(stat_data['waveDirectionBins']))
    frqbin = fid.createDimension('waveFrequency', np.size(stat_data['waveFrequency']))
    
    #
    # convert to Lat/lon here

    # write data to the nc file
    write_data_to_nc(fid, stat_var_atts, stat_data)
    # close file
    fid.close()

def convert_FRFgrid(gridFname, ofname, globalYaml, varYaml, plotFlag=False):
    """This function will convert the FRF gridded text product into a NetCDF file

    Args:
      gridFname: input FRF gridded product
      ofname: output netcdf filename
      globalYaml: a yaml file containing global meta data
      varYaml: a yaml file containing variable meta data
      plotFlag: true or false for creation of QA plots (Default value = False)

    Returns:
      None

    """
    # Defining rigid parameters
    raise NotImplementedError('is this depricated?')
    # defining the bounds of the FRF gridded product
    gridYmax = 1100  # maximum FRF Y distance for netCDF file
    gridYmin = -100  # minimum FRF Y distance for netCDF file
    gridXmax = 950  # maximum FRF X distance for netCDF file
    gridXmin = 50  # minimum FRF xdistance for netCDF file
    fill_value= '-999.0'
    # main body
    # load Grid from file
    tempClass = PrepData.inputOutput.genericIO()
    xyz = tempClass.importXYZ(gridFname)

    # make dictionary in right form
    dx = np.median(np.diff(xyz['x']))
    dy = np.max(np.diff(xyz['y']))
    xgrid = np.unique(xyz['x'])
    ygrid = np.unique(xyz['y'])

    # putting the loaded grid into a 2D array
    zgrid = np.zeros((len(xgrid), len(ygrid)))
    rc = 0
    for i in range(np.size(ygrid, axis=0 )):
        for j in range(np.size(xgrid, axis=0)):
            zgrid[j, i] = xyz['z'][rc]
            rc += 1
    if plotFlag == True:
        from matplotlib import pyplot as plt
        plt.pcolor(xgrid, ygrid, zgrid.T)
        plt.colorbar()
        plt.title('FRF GRID %s' % ofname[:-3].split('/')[-1])
        plt.savefig(ofname[:-4] + '_RawGridTxt.png')
        plt.close()
    # aking labels in FRF coords for
    ncXcoord = np.linspace(gridXmin, gridXmax, num=(gridXmax - gridXmin) / dx + 1, endpoint=True)
    ncYcoord = np.linspace(gridYmin, gridYmax, num=(gridYmax - gridYmin) / dy + 1, endpoint=True)
    frame = np.full((np.shape(ncXcoord)[0], np.shape(ncYcoord)[0]), fill_value=fill_value)

    # find the overlap locations between grids
    xOverlap = np.intersect1d(xgrid, ncXcoord)
    yOverlap = np.intersect1d(ygrid, ncYcoord)
    assert len(yOverlap) >= 3, 'The overlap between grid nodes and netCDF grid nodes is short'
    lastX = np.argwhere(ncXcoord == xOverlap[-1])[0][0]
    firstX = np.argwhere(ncXcoord == xOverlap[0])[0][0]
    lastY = np.argwhere(ncYcoord == yOverlap[-1])[0][0]
    firstY = np.argwhere(ncYcoord == yOverlap[0])[0][0]

    # fill the frame grid with the loaded data
    frame[firstX:lastX+1, firstY:lastY+1] = zgrid

    # run data check
    assert set(xOverlap).issubset(ncXcoord), 'The FRF X values in your function do not fit into the netCDF format, please rectify'
    assert set(yOverlap).issubset(ncYcoord), 'The FRF Y values in your function do not fit into the netCDF format, please rectify'

    # putting the data into a dictioary to make a netCDF file
    fields = gridFname.split('_')
    for fld in fields:
        if len(fld) == 8:
            dte = fld  # finding the date in the file name
            break
    gridDict = {'zgrid': frame,
                'xgrid': ncXcoord,
                'ygrid': ncYcoord,
                'time': nc.date2num(DT.datetime(int(dte[:4]), int(dte[4:6]),
                                                       int(dte[6:])), 'seconds since 1970-01-01')}

    # making the netCDF file from the gridded data
    makenc_FRFGrid(gridDict, ofname, globalYaml, varYaml)

def makeDirectionalWavesWHOI(ofname, dataDict, globalYaml, varYaml):
    """
    Args:
      ofname: output file name
      dataDict: input data dictionary matching variable names in yaml
      globalYaml: global yaml for meta data ahead of file
      varYaml: variable data structured the same to that of the dataDict

    Returns:
      None  - writes out the netCDF file

    """
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)
    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    #### create dimensions
    tdim = fid.createDimension('time', np.shape(dataDict['time'])[0])  # None = size of the dimension, what does this gain me if i know it
    statnamelen = fid.createDimension('station_name_length', len(dataDict['station_name']))
    dirbin = fid.createDimension('waveDirectionBins', np.size(dataDict['directionBands']))
    frqbin = fid.createDimension('waveFrequency', np.size(dataDict['freqBands']))

    # write data to the nc file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_todaysBathyCMTB(gridDict, ofname, globalYaml, varYaml):
    """Generate bathymetry file for CMTB
    Args:
      gridDict: data dictionary matching varYaml
      ofname: file output name
      globalYaml: yaml containing CF compliant meta data
      varYaml: yaml containing matching data structure to gridDict and CF compliant meta data

    Returns:

    """
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xFRF = fid.createDimension('xFRF', gridDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', gridDict['yFRF'].shape[0])
    time = fid.createDimension('time', np.size(gridDict['time']))
    # write data to file
    write_data_to_nc(fid, varAtts, gridDict)
    # close file
    fid.close()

def makenc_CSHORErun(ofname, dataDict, globalYaml, varYaml):
    """This is a function that makes netCDF files from CSHORE model runs created by
       David Young using all the stuff Spicer Bak used. You have to build dataDict from the different dictionaries
       output by cshore_io.load_CSHORE_results().  YOU DONT HAVE TO HAND IT LAT LON THOUGH!!!

    Args:
      dataDict: keys:
        time: - time steps of the simulation nc file

        xFRF: - xFRF positions of the simulation

        aveE: - depth averaged eastward current!

        stdE: - standard deviation of eastward current

        aveN: - same as above but northward current

        stdN: - same as above but northward

        waveHs: - significant wave heights

        waveMeanDirection:  mean direction of the waves at each cross-shore position

        waterLevel: mean water level at each cross-shore position

        stdWaterLevel: standard deviation of the water surface elevation at each cross-shore position

        setup: wave setup at each cross-shore position

        runup2perc: 2 percent exceedance runup elevation for each model time-step

        runupMean: mean runup elevation for each model time-step

        qbx: cross-shore bed load sediment transport rate

        qsx: cross-shore suspended sediment transport rate

        qby: alongshore bed load sediment transport rate

        qsy: alongshore suspended sediment transport rate

        probabilitySuspension: probability that sediment will be suspended at particular node

        probabilityMovement: probability that sediment will move

        suspendedSedVolume: suspended sediment volume at each cross-shore position

        bottomElevation: the bottom elevation at each xFRF position in the simulation

        surveyNumber:  this is the surveyNumber that the integrated bathymetry for this simulation was built on

        profileNumber: this is either the profileNumber of the survey or the alongshore position of the integratred bathymetry transect that is used as the bed elevation boundary condition

        bathymetryDate: this is the day that the aforementioned survey was taken

        yFRF: this is the yFRF position of the transect itself.  if it is the integrated bathymetry, then this will be identical to the profileNumber

      ofname (str): this is the FULL PATH INCLUDING FILENAME AND EXTENSION to the position where the ncFile will be saved when output
      globalYaml (str): full path to the globalYaml used to build this ncFile
      varYaml (str): full path to the variableYaml used to build this ncFile

    Returns:
      netCDF file with CSHORE model results in it

    """
    from testbedutils import geoprocess as gp  # this might create a circular import
    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # note: you have to hand this the yFRF coordinates of the BC gage if you want to get lat/lon..
    lx = np.size(dataDict['xFRF'], axis=0)
    lat = np.zeros(lx)
    lon = np.zeros(lx)
    for ii in range(0, lx):
        coords = gp.FRFcoord(dataDict['xFRF'][ii], dataDict['yFRF'])
        lat[ii] = coords['Lat']
        lon[ii] = coords['Lon']
    dataDict['latitude'] = lat
    dataDict['longitude'] = lon

    # ok, we are HARD CODING the dimensions to ALWAYS be at the 8m ARRAY (xFRF = 914.44 rounded DOWN to 914)
    # we will just fill in the missing values with nans as required
    array8m_loc = 914

    # creating dimensions of data
    new_s = np.shape(range(-50, array8m_loc+1))[0]
    new_t = np.shape(dataDict['waveHs'])[0]
    xFRF = fid.createDimension('xFRF', new_s)
    time = fid.createDimension('time', new_t)

    # check to see if the grid I am importing is smaller than my netCDF grid
    if np.shape(range(-50, array8m_loc+1))[0] == np.shape(dataDict['xFRF']):
        # the model grid is the same as the netCDF grid, so do nothing
        dataDict_n = dataDict
        pass
    else:
        dataDict_n = {'xFRF': np.flipud(np.array(range(-50, array8m_loc+1)) + 0.0),
                      'time': dataDict['time'],
                      'aveE': np.full((new_t, new_s), fill_value=np.nan),
                      'stdE': np.full((new_t, new_s), fill_value=np.nan),
                      'aveN': np.full((new_t, new_s), fill_value=np.nan),
                      'stdN': np.full((new_t, new_s), fill_value=np.nan),
                      'waveHs': np.full((new_t, new_s), fill_value=np.nan),
                      'waveMeanDirection': np.full((new_t, new_s), fill_value=np.nan),
                      'waterLevel': np.full((new_t, new_s), fill_value=np.nan),
                      'stdWaterLevel': np.full((new_t, new_s), fill_value=np.nan),
                      'setup': np.full((new_t, new_s), fill_value=np.nan),
                      'runup2perc': dataDict['runup2perc'],
                      'runupMean': dataDict['runupMean'],
                      'qbx': np.full((new_t, new_s), fill_value=np.nan),
                      'qsx': np.full((new_t, new_s), fill_value=np.nan),
                      'qby': np.full((new_t, new_s), fill_value=np.nan),
                      'qsy': np.full((new_t, new_s), fill_value=np.nan),
                      'probabilitySuspension': np.full((new_t, new_s), fill_value=np.nan),
                      'probabilityMovement': np.full((new_t, new_s), fill_value=np.nan),
                      'suspendedSedVolume': np.full((new_t, new_s), fill_value=np.nan),
                      'bottomElevation': np.full((new_t, new_s), fill_value=np.nan),
                      'latitude': np.full((new_s), fill_value=np.nan),
                      'longitude': np.full((new_s), fill_value=np.nan),
                      'surveyNumber': dataDict['surveyNumber'],
                      'profileNumber': dataDict['profileNumber'],
                      'bathymetryDate': dataDict['bathymetryDate'],
                      'yFRF': dataDict['yFRF'],}

        if 'FIXED' in ofname:
            dataDict_n['bottomElevation'] = np.full((new_s), fill_value=np.nan)
        elif 'MOBILE' in ofname:
            dataDict_n['bottomElevation'] = np.full((new_t, new_s), fill_value=np.nan)
        else:
            print 'You need to modify makenc_CSHORErun in makenc.py to accept your new version name!'

        # find index of first point on dataDict grid
        min_x = min(dataDict['xFRF'])
        ind_minx = int(np.argwhere(dataDict_n['xFRF'] == min_x))
        max_x = max(dataDict['xFRF'])
        ind_maxx = int(np.argwhere(dataDict_n['xFRF'] == max_x))

        for ii in range(0, int(new_t)):
            dataDict_n['aveE'][ii][ind_maxx:ind_minx+1] = dataDict['aveE'][ii]
            dataDict_n['stdE'][ii][ind_maxx:ind_minx+1] = dataDict['stdE'][ii]
            dataDict_n['aveN'][ii][ind_maxx:ind_minx+1] = dataDict['aveN'][ii]
            dataDict_n['stdN'][ii][ind_maxx:ind_minx+1] = dataDict['stdN'][ii]
            dataDict_n['waveHs'][ii][ind_maxx:ind_minx+1] = dataDict['waveHs'][ii]
            dataDict_n['waveMeanDirection'][ii][ind_maxx:ind_minx+1] = dataDict['waveMeanDirection'][ii]
            dataDict_n['waterLevel'][ii][ind_maxx:ind_minx+1] = dataDict['waterLevel'][ii]
            dataDict_n['stdWaterLevel'][ii][ind_maxx:ind_minx+1] = dataDict['stdWaterLevel'][ii]
            dataDict_n['setup'][ii][ind_maxx:ind_minx+1] = dataDict['setup'][ii]
            dataDict_n['qbx'][ii][ind_maxx:ind_minx+1] = dataDict['qbx'][ii]
            dataDict_n['qsx'][ii][ind_maxx:ind_minx+1] = dataDict['qsx'][ii]
            dataDict_n['qby'][ii][ind_maxx:ind_minx+1] = dataDict['qby'][ii]
            dataDict_n['qsy'][ii][ind_maxx:ind_minx+1] = dataDict['qsy'][ii]
            dataDict_n['probabilitySuspension'][ii][ind_maxx:ind_minx+1] = dataDict['probabilitySuspension'][ii]
            dataDict_n['probabilityMovement'][ii][ind_maxx:ind_minx+1] = dataDict['probabilityMovement'][ii]
            dataDict_n['suspendedSedVolume'][ii][ind_maxx:ind_minx+1] = dataDict['suspendedSedVolume'][ii]
            dataDict_n['latitude'][ind_maxx:ind_minx+1] = dataDict['latitude'][ii]
            dataDict_n['longitude'][ind_maxx:ind_minx+1] = dataDict['longitude'][ii]


        if 'FIXED' in ofname:
            dataDict_n['bottomElevation'][ind_maxx:ind_minx + 1] = dataDict['bottomElevation']
        elif 'MOBILE' in ofname:
            for ii in range(0, int(new_t)):
                dataDict_n['bottomElevation'][ii][ind_maxx:ind_minx + 1] = dataDict['bottomElevation'][ii]
        else:
            print 'You need to modify makenc_CSHORErun in makenc.py to accept your new version name!'

    # get rid of all masks
    test = np.ma.masked_array(dataDict_n['aveE'], np.isnan(dataDict_n['aveE']))
    dataDict_n['aveE'] = test
    del test
    test = np.ma.masked_array(dataDict_n['stdE'], np.isnan(dataDict_n['stdE']))
    dataDict_n['stdE'] = test
    del test
    test = np.ma.masked_array(dataDict_n['aveN'], np.isnan(dataDict_n['aveN']))
    dataDict_n['aveN'] = test
    del test
    test = np.ma.masked_array(dataDict_n['stdN'], np.isnan(dataDict_n['stdN']))
    dataDict_n['stdN'] = test
    del test
    test = np.ma.masked_array(dataDict_n['waveHs'], np.isnan(dataDict_n['waveHs']))
    dataDict_n['waveHs'] = test
    del test
    test = np.ma.masked_array(dataDict_n['waveMeanDirection'], np.isnan(dataDict_n['waveMeanDirection']))
    dataDict_n['waveMeanDirection'] = test
    del test
    test = np.ma.masked_array(dataDict_n['waterLevel'], np.isnan(dataDict_n['waterLevel']))
    dataDict_n['waterLevel'] = test
    del test
    test = np.ma.masked_array(dataDict_n['stdWaterLevel'], np.isnan(dataDict_n['stdWaterLevel']))
    dataDict_n['stdWaterLevel'] = test
    del test
    test = np.ma.masked_array(dataDict_n['setup'], np.isnan(dataDict_n['setup']))
    dataDict_n['setup'] = test
    del test
    test = np.ma.masked_array(dataDict_n['qbx'], np.isnan(dataDict_n['qbx']))
    dataDict_n['qbx'] = test
    del test
    test = np.ma.masked_array(dataDict_n['qsx'], np.isnan(dataDict_n['qsx']))
    dataDict_n['qsx'] = test
    del test
    test = np.ma.masked_array(dataDict_n['qby'], np.isnan(dataDict_n['qby']))
    dataDict_n['qby'] = test
    del test
    test = np.ma.masked_array(dataDict_n['qsy'], np.isnan(dataDict_n['qsy']))
    dataDict_n['qsy'] = test
    del test
    test = np.ma.masked_array(dataDict_n['probabilitySuspension'], np.isnan(dataDict_n['probabilitySuspension']))
    dataDict_n['probabilitySuspension'] = test
    del test
    test = np.ma.masked_array(dataDict_n['probabilityMovement'], np.isnan(dataDict_n['probabilityMovement']))
    dataDict_n['probabilityMovement'] = test
    del test
    test = np.ma.masked_array(dataDict_n['suspendedSedVolume'], np.isnan(dataDict_n['suspendedSedVolume']))
    dataDict_n['suspendedSedVolume'] = test
    del test
    test = np.ma.masked_array(dataDict_n['latitude'], np.isnan(dataDict_n['latitude']))
    dataDict_n['latitude'] = test
    del test
    test = np.ma.masked_array(dataDict_n['longitude'], np.isnan(dataDict_n['longitude']))
    dataDict_n['longitude'] = test
    del test
    test = np.ma.masked_array(dataDict_n['bottomElevation'], np.isnan(dataDict_n['bottomElevation']))
    dataDict_n['bottomElevation'] = test
    del test

    # check to see if I screwed up!
    assert set(dataDict.keys()) == set(dataDict_n.keys()), 'You are missing dictionary keys in the new dictionary!'
    # replace the dictionary with the new dictionary
    del dataDict
    dataDict = dataDict_n
    del dataDict_n

    # now we flip everything that has a spatial dimension around so it will be all pretty like spicer wants?
    dataDict['aveN'] = np.flip(dataDict['aveN'], 1)
    dataDict['waveHs'] = np.flip(dataDict['waveHs'], 1)
    dataDict['aveE'] = np.flip(dataDict['aveE'], 1)
    dataDict['waveMeanDirection'] = np.flip(dataDict['waveMeanDirection'], 1)
    dataDict['stdWaterLevel'] = np.flip(dataDict['stdWaterLevel'], 1)
    dataDict['probabilitySuspension'] = np.flip(dataDict['probabilitySuspension'], 1)
    dataDict['stdN'] = np.flip(dataDict['stdN'], 1)
    dataDict['stdE'] = np.flip(dataDict['stdE'], 1)
    dataDict['bottomElevation'] = np.flip(dataDict['bottomElevation'], 1)
    dataDict['xFRF'] = np.flip(dataDict['xFRF'], 0)
    dataDict['qsy'] = np.flip(dataDict['qsy'], 1)
    dataDict['qsx'] = np.flip(dataDict['qsx'], 1)
    dataDict['waterLevel'] = np.flip(dataDict['waterLevel'], 1)
    dataDict['qbx'] = np.flip(dataDict['qbx'], 1)
    dataDict['qby'] = np.flip(dataDict['qby'], 1)
    dataDict['setup'] = np.flip(dataDict['setup'], 1)
    dataDict['longitude'] = np.flip(dataDict['longitude'], 0)
    dataDict['latitude'] = np.flip(dataDict['latitude'], 0)
    dataDict['suspendedSedVolume'] = np.flip(dataDict['suspendedSedVolume'], 1)
    dataDict['probabilityMovement'] = np.flip(dataDict['probabilityMovement'], 1)

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_intBATHY(ofname, dataDict, globalYaml, varYaml):
    """
    TODO: can this be combined with makenc_t0BATHY
    Args:
      ofname: this is the name of the cshore_ncfile you are building
      dataDict: keys must include... and match the varYaml file
        utmNorthing - this is utm in meters (not feet)

        utmEasting - this is utm in meters (not feet)
    
      globalYaml: yaml containing global meta data
      varYaml:  yaml containing variable meta data

    Returns:
      writes out the ncfile

    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    ni = fid.createDimension('ni', dataDict['utmEasting'].shape[1])
    nj = fid.createDimension('nj', dataDict['utmEasting'].shape[0])

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_t0BATHY(ofname, dataDict, globalYaml, varYaml):
    """# this is the script that builds the t0 netCDF file from the initial Bathy DEM (intBathy)

    Args:
      ofname (str): this is the name of the cshore_ncfile you are building
      dataDict (dict): keys must include... and matching keys to var Yaml
        xFRF - in m

        yFRF - in m

      globalYaml (str): CF compliant meta data in global file
      varYaml (str):  CF compliant variable meta data matching dataDict

    Returns:
      writes out the cshore_ncfile

    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    xFRF = fid.createDimension('xFRF', dataDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', dataDict['yFRF'].shape[0])

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()

def makenc_tiBATHY(ofname, dataDict, globalYaml, varYaml):
    """# this is the script that builds the monthly ti netCDF file by incorporating the new survey data into the most recent Bathy DEM

    Args:
      ofname (str): this is the name of the cshore_ncfile you are building
      dataDict (str): keys must include... and matching keys to varYaml
            time:
            xFRF: - in m
            yFRF: - in m

      globalYaml (str): path to global yaml with CF compliant meta data
      varYaml (str): path to variable CF compliant meta data and matching the keys in data Dict

    Returns:
      writes out the netCDF file

    """

    globalAtts = import_template_file(globalYaml)
    varAtts = import_template_file(varYaml)

    # create netcdf file
    fid = init_nc_file(ofname, globalAtts)

    # creating dimensions of data
    time = fid.createDimension('time', dataDict['time'].shape[0])
    xFRF = fid.createDimension('xFRF', dataDict['xFRF'].shape[0])
    yFRF = fid.createDimension('yFRF', dataDict['yFRF'].shape[0])

    # write data to file
    write_data_to_nc(fid, varAtts, dataDict)
    # close file
    fid.close()


