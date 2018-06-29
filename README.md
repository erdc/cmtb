# README #

This is the official repository for the FRF's Coastal Model Test Bed in which numerical models are run in (near) real-time for testing and evaluation of the models.
The test bed is established at the USACE CHL Field REsearch Facility in Duck, North Carolina.  Any contributions are welcome! 

### What is this repository for? ###

* This repository is to help facilitate the incorporation of new models
* There are various scripts that will
    - Run models in a operational environment
    - prep FRF specific data into model expected conventions
    - read/write data from/to model specific output/input files
    - log model data to netCDF files and create plots for model comparisons

* Models Running:
    - STWAVE 
    - CSHORE
    - CMS wave
    - CMS Flow development
* analysis scripts available:
    - basic statistics to plot on daily runs
    - developing standard analysis routines for long term analysis (hand url)
    - sea and swell separation could be improved
    

### How do I get set up? ###

* Read the API
* Reach out about including a new model (create an issue)


#### Python Packages Required

  * submodule dependencies (community Developed)
    - testbedutils  - utility package, including wave and geoprocessing 
    - prepdata - this holds model io packages and data preparation routines
    - getdatatestbed - operates - built with numpy 1.13 (at least)
  
  * python packages required
    - netCDF4
    - pyproj
    - utm
    These can be installed by
    - pip install <package>
    - conda install <package>
  
  - there's likely more, please add (or let me know) as you find!!  :-[]
  
### What is the structure?

* start with Runwork flow _ stwave or CMS
    - NOTES:
        * STWAVE is the original and is completely functional 
        * CMS is a little more elegant on the post processing side (but currently not functional)
        * This framework built with a preprocessing script and a post processing script   
    - __Pre-Processing__
        1. get data (initalized with date time instances)
        2. get data . [wind, waves, water level, bathy]
            * each get data is returned in a dictionary with specific keys
        3. each get data dictionary is preped for model input 
            * vector, scalar averaged for winds and water level to time match that of waves 
            * wave data are rotated shore normal and flipped coordinate (LH -> RH)
        4. data are input to an instance of prepdata.inputoutput  [cms/stwave]
            * each one of writes out the appropriate model input files (specific to each model) 
    - __Model Is run__
    - __Post-processing__
        * output files are parsed into data dictionaries (similar to get data)
        * model data are massaged 
            * any data that was done in i.c. is undone here (wave angles rotated to True north)
        * plots are made from rotated files, statistics are calculated and put in the plots
        * data are logged to netCDF and transfered to local/remote CHL Thredds location
        
### Assumptions 

There are general assumptions that are made to make the system work, they are listed below:
* A sub folder in the project root directory ['cmtb'] is created with the name and has the following structure
    * data/[wave, circultation, morphology]/[model name]/[date of simulation]/[files for simulstions & figures directory] 
    * The above is standard, but can be modified with newly developed input file
* python packages are installed:
    - netCDF4 - conda install
    - pyproj - pip install 
    - utm - conda install (maybe pip)
* netCDF ouput are delievered to the thredds_data folder parallel to the CMTB folder in the users home directory the structure is similar to that of the data folder in the CMTB folder 
    
### Contribution guidelines 

The code is written and run in 2.7, but code should be written with python3 conventions to facilitate the migration from 2->3 in the next few years.
Contributions are reviewed by administrators for the project.  Forking and pull requests are the preferred method for project contributions. 
Please document your code with established doc-string conventions and in the sphinx environment as it is setup moving forward.  

Issues that identified and need resolution as improvement proceeds: 
* general Model Tests:
    - examples: test wind generation by blowing wind over flat sloped bathymetry... etc
* make gauge observation locations selections more generalized
    - currently gauge locations are queried from the ncml, which isn't entirely correct as gauges have moved through the history though have stayed constant since 2015.
    - suggested fix: create look up table from gague locations pulled directly from each netCDF file.  then based on dates of sim, use lookup table to determine what gauges to use
    - This could be more robustly quereied given a date range for the sim setup, a thredds crawler could be used to look for all dates/times 

* a more universal wrapper script could be used for all models or setups 
* a working example setup could be created to demonstrate the work flow 
* visualization and anaylsis could be more universalized
* Code contributions review and incorporation could be better established --- currently me!  

