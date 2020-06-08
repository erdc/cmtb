from getdatatestbed import getDataFRF
import datetime as DT
import xarray as xr
from matplotlib import pyplot as plt
import glob
import numpy as np
from scipy import interpolate
import pandas as pd
import pickle

start = DT.datetime(2018, 2, 27)
end =   DT.datetime(2018, 3, 20,3)
gd  =   getDataFRF.getDataTestBed(start, end)
bathy = gd.getBathyIntegratedTransect(ybounds=[940, 950], forceReturnAll=True)

# go = getDataFRF.getObs(start, end)
# topo = go.getLidarTopo(ybounds=[940, 950])
topoURL1 = 'https://chldata.erdc.dren.mil/thredds/dodsC/frf/geomorphology/DEMs/duneLidarDEM/2018/FRF-geomorphology_DEMs_duneLidarDEM_201802.nc'
topoURL2 = "https://chldata.erdc.dren.mil/thredds/dodsC/frf/geomorphology/DEMs/duneLidarDEM/2018/FRF-geomorphology_DEMs_duneLidarDEM_201803.nc"
# febTopo_nc = xr.load_dataset(topoURL1)
lidarFiles = sorted(glob.glob('lidarTopoData/*/FRF-geomorphology_DEMs_duneLidarDEM*.nc'))


plt.figure()
for tt in range(bathy['elevation'].shape[0]):
    Yidx = bathy['yFRF']==945
    plt.plot(bathy['xFRF'], bathy['elevation'][tt, Yidx].squeeze(), '--', label='bathy {}'.format(bathy['time'][tt]))
for topof in lidarFiles:
    topo = xr.load_dataset(topof)
    Yidx = topo['yFRF'] == 945
    plt.plot(topo['xFRF'], topo.elevation[:,Yidx].squeeze(), '-', label='topo {}'.format(topo.time.to_dict()[
                                                                                              'data'][0]))
    
plt.legend()
plt.xlim([40, 150])
plt.ylim([-1, 8])
yLoc = 945
slopeVal, ms, saveBathy = 5, 5, {}
for tt in range(bathy['elevation'].shape[0]):
    for topof in lidarFiles:
        #load bathy
        origBathyProfile = bathy['elevation'][tt, bathy['yFRF']==yLoc].squeeze()
        origbathyX = bathy['xFRF']
        # load topo
        topo = xr.load_dataset(topof)
        topoProfile = topo.elevation[:, topo['yFRF']==yLoc].squeeze()
        topoX = topo.xFRF[topoProfile.notnull()]
        topoProfile = topoProfile[topoProfile.notnull()]
        #truncate bathy
        idxNearestBathy = np.argwhere((bathy['xFRF'] - topoX[topoProfile.notnull()].max()) > 0).min()
        bathyProfile = origBathyProfile[idxNearestBathy+slopeVal:]
        bathyX = origbathyX[idxNearestBathy+slopeVal:]
        
        # interpolate all topo to truncated bathy with 1D cubic spline
        interpf = interpolate.interp1d(np.append(topoX, bathyX), np.append(topoProfile, bathyProfile), kind='cubic',
                                       fill_value='extrapolate')
        cubicX = np.arange(topoX.min(), bathyX.max())
        cubicProfile = interpf(cubicX)
        
        #now plot to see what it looks like
        topoDate = pd.to_datetime(topo.time.data[0]).strftime('%Y-%m-%d')
        bathyDate = bathy['time'][tt].strftime('%Y-%m-%d')
        title = 'fuzed topo {} to bathy {}'.format(topoDate, bathyDate)
        plt.figure(figsize=(12,4));
        plt.plot(topoX, topoProfile, 'g.', ms =ms, label='original Topo')
        plt.plot(origbathyX, origBathyProfile, 'r.', ms=ms, label='original Bathy')
        plt.plot(bathyX, bathyProfile, 'b.', ms=ms, label='input bathy')
        plt.plot(cubicX, cubicProfile, label='cubicProfile')
        # plt.plot(bathyX, bathyProfile, label='interp input bathy')
        plt.legend()
        plt.xlim([50, 150])
        plt.ylim([-1, 7])
        plt.title(title)
        plt.ylabel('elevation')
        plt.xlabel('xFRF')
        plt.savefig('bathyFusion_{}to{}.png'.format(topoDate, bathyDate))
        key = "t_{}_b{}".format(topoDate, bathyDate)
        saveBathy[key] = cubicProfile
        
        # now save
        out = {'xFRF':      cubicX,
               'yFRF':      yLoc,
               'elevation': cubicProfile,
               'time':      DT.datetime.strptime(topoDate, '%Y-%m-%d'),
               'lat':       -999,
               'lon':       -999}
        outfname = "bathyPickle_{}.pickle".format(topoDate)
        with open(outfname, 'wb') as fid:
            pickle.dump(out, fid,  protocol=pickle.HIGHEST_PROTOCOL)
        
###### test spline method code
origBathyProfile = bathy['elevation'][tt, bathy['yFRF']==yLoc].squeeze()
origbathyX = bathy['xFRF']
# load topo
topof = lidarFiles[1]
topo = xr.load_dataset(topof)
topoProfile = topo.elevation[:, topo['yFRF']==yLoc].squeeze()
topoX = topo.xFRF[topoProfile.notnull()]
topoProfile = topoProfile[topoProfile.notnull()]
#truncate bathy
slopeVal = 5  # cell count to eliminate
idxNearestBathy = np.argwhere((bathy['xFRF'] - topoX[topoProfile.notnull()].max()) > 0).min()
bathyProfile = origBathyProfile[idxNearestBathy+slopeVal:]
bathyX = origbathyX[idxNearestBathy+slopeVal:]
# order = 3//
# knots = zip(np.append(topoX, bathyX), np.append(topoProfile, bathyProfile))
# coeffs = np.ones_like(knots)
# out = interpolate.BSpline(knots, coeffs, order)

interpf = interpolate.interp1d(np.append(topoX, bathyX), np.append(topoProfile, bathyProfile), kind='cubic',
                               fill_value='extrapolate')
cubicX = np.arange(topoX.min(), bathyX.max())
cubicProfile = interpf(cubicX)

ms = 4
plt.figure();
plt.plot(topoX, topoProfile, 'g.', ms =ms, label='original Topo')
plt.plot(origbathyX, origBathyProfile, 'r.', ms=ms, label='original Bathy')
plt.plot(bathyX, bathyProfile, 'b.', ms=ms, label='input bathy')
plt.plot(cubicX, cubicProfile, label='cubicProfile')
# plt.plot(bathyX, bathyProfile, label='interp input bathy')
plt.legend()
plt.xlim([50, 120])
plt.ylim([-1, 7])