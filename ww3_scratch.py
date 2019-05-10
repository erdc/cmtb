from prepdata import inputOutput
from getdatatestbed import getDataFRF
import matplotlib.pyplot as plt
import meshio as mio
import datetime as DT
from matplotlib import tri

ww3io = inputOutput.ww3IO('')
start = DT.datetime(2018, 1, 2)
end = DT.datetime(2018, 2,1)

fname = '/home/spike/cmtb/grids/ww3/Mesh_interp_Mar2018.msh'
data = mio.read(fname)
## data. points are lon, lat, elevations from below
gdtb = getDataFRF.getDataTestBed(start, end, THREDDS='CHL')
newBathy = gdtb.getBathyIntegratedTransect()
# load background data
data.points
triObj = tri.Triangulation(data.points[0], data.points[1])
ftri = tri.LinearTriInterpolator(triObj, data.points[2])
newZs = ftri(newBathy['lon'], newBathy['lat'])

# the old school way
data2 = ww3io.load_msh(fname)



data.points
plt.tripcolor(data['lon'], data['lat'], data['elevations'])
plt.colorbar()
plt.show()


meshio.write_points_cells('myfile.msh', data.points, data.cells)