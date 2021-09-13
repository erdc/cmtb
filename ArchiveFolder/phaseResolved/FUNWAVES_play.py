from getdatatestbed import getDataFRF
import datetime as DT
from prepdata import prepDataLib as pdl
from testbedutils import sblib as sb
import numpy as np
## getting data for matt and gabby to write input files
pd = pdl.PrepDataTools() # initalize for WL processing

start = DT.datetime(2018, 10, 3, 17, 00)
start1 = DT.datetime(2018, 10, 5, 17, 00)
end = DT.datetime(2018, 10, 5, 18, 00)

go1 = getDataFRF.getObs(start, start1, THREDDS='CHL')
go2 = getDataFRF.getObs(start1, end, THREDDS='CHL')

WL1 = go1.getWL()
WL2 = go2.getWL()
WL1 = pd.prep_WL(WL1, [DT.datetime(2018, 10, 4, 17)])
WL2 = pd.prep_WL(WL2, [DT.datetime(2018, 10, 5, 17)])
import netCDF4 as nc
ncfile = nc.Dataset('https://chlthredds.erdc.dren.mil/thredds/dodsC/frf/geomorphology/elevationTransects/survey/FRF_20181004_1155_FRF_NAVD88_LARC_GPS_UTC_v20181012.nc')
bathy = {'elevation': ncfile['elevation'][:],
        'time': nc.num2date(ncfile['time'][:], 'seconds since 1970-01-01'),
        'xFRF': ncfile['xFRF'][:],
        'yFRF': ncfile['yFRF'][:],}
# bathy1 = go1.getBathyTransectFromNC()
wave1 = go1.getWaveSpec('8m-array')
wave2 = go2.getWaveSpec('8m-array')

# now merge to one dictionary
def mergeDicts(dict1, dict2):

    ds = [dict1, dict2]
    d = {}
    for k in dict1.keys():
      d[k] = list(d[k] for d in ds)
    return d
WL = mergeDicts(WL1, WL2)

idx1 = np.argwhere(wave1['time'] == DT.datetime(2018, 10, 4, 17)).squeeze()
idx2 = np.argwhere(wave2['time'] == DT.datetime(2018, 10, 5, 17)).squeeze()
wave1 = sb.reduceDict(wave1, idx1)
waves = mergeDicts(wave1, wave2)
import glob
fname = glob.glob('/home/spike/Downloads/*.nc')
toponcfile = nc.Dataset(fname[0])
topo1 = {'elevation': toponcfile['elevation'][:],
         'xFRF': toponcfile['xFRF'][:],
         'yFRF': toponcfile['yFRF'][:]}
toponcfile = nc.Dataset(fname[1])
topo2 = {'elevation': toponcfile['elevation'][:],
         'xFRF': toponcfile['xFRF'][:],
         'yFRF': toponcfile['yFRF'][:]}
topo = mergeDicts(topo1, topo2)
import pickle
fileOut = 'MattsFile.pickle'
with open(fileOut, 'wb') as fid:
    pickle.dump((waves, WL, bathy, topo), fid, protocol=pickle.HIGHEST_PROTOCOL)

# to open

with open(fileOut, 'rb') as fid:
    data = pickle.load(fid)

