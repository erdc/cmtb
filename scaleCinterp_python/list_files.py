# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 14:28:56 2014

@author: jwlong
"""
import glob
def list_files(datapath, datatype):
    """

    Args:
      datapath: param datatype:
      datatype: 

    Returns:

    """
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    # filelist = []
    # listing = os.listdir(datapath)
    # for files in listing:
    #     if files.endswith(datatype):
    #        filelist.append(files)
    # return filelist
    return glob.glob(datapath+'*'+datatype)  # i think this should do the same thing as above
