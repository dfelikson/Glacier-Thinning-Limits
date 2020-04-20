#!/usr/bin/env python
import numpy as np
from glacier import glacier

from matplotlib import pyplot as plt

centerlineFilename = 'centerlines/flowline0001.shp'
glacierName = 'Jakobshavn'

# 
# NOTE: Uncertainty can be specified as either spatially-varying (raster) or as a constant but it must be provided
#       because the error is propagated through the Peclet number calculation.
bedFilename = 'data/bed.tif'
bedErrFilename = 'data/errbed.tif'
surfFilename = 'data/surface.tif'
surfError = 10.

# NOTE: The convention I use is that all input files (bed and surface) are referenced to the ellipsoid. One of the 
#       flow laws that can be used in the Peclet number calculation requires height above sea-level. So, I read in
#       the geoid and use that as local sea level.
geoidFilename = 'data/geoid.tif'

# --- Instantiate -------
print('instantiating object')
gl = glacier(centerlineFilename, centerline_sample_distance=50., interpDistThreshold=2000.0)
gl.name = glacierName

# --- Load datasets -------
print('loading datasets')

# Bed
print(' -> bed')
gl.new_dataset(name='bed', type='raster', filename=bedFilename, uncertainty=bedErrFilename, nodataValues=-9999)

# Surface
print(' -> surface')
gl.new_dataset(name='surface', type='raster', filename=surfFilename, uncertainty=surfError, nodataValues=[-9999.0, 0.0])

# Geoid
gl.new_dataset(name='Geoid', type='raster', filename=geoidFilename, nodataValues=-32767)
sea_level = gl.datasets['Geoid'].get_z(0)

# NOTE: Let's plot the glacier geometry for a quick check
plt.plot(gl.centerline.distance, gl.datasets['bed'].get_z(), 'k.-')
plt.plot(gl.centerline.distance, gl.datasets['surface'].get_z(), 'b.-')
plt.show()

# --- Peclet number -------
print('calculating Peclet number')

gl.new_dataset('Pe')
# NOTE: The following parameter specifies how the geometry is smoothed before the Peclet number calculation. It is
#       the number of ice thicknesses over which the geometry is averaged.
gl.datasets['Pe'].moving_average_thickness_factor = 10.
gl.Pe('bed', 'surface', 'Pe', sea_level=sea_level)

# NOTE: Now, let's plot the Peclet number. Note that we need to use the "get_zmvavg" method here. Because the Peclet
#       number comes from a smoothed geometry, I populate the zmvavg array instead of z for the Peclet dataset.
plt.plot(gl.centerline.distance, gl.datasets['Pe'].get_zmvavg(), 'k.-')
plt.show()

