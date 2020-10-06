# Glacier-Thinning-Limits
The Glacier Thinning Limits tool can be used to calculate the Peclet number along glacier flowlines, that describes how terminus-initiated thinning will evolve. The code here accompanies [Felikson et al. (2017)](https://doi.org/10.1038/ngeo2934).

To get started, execute the run_glacier.py script. This requires the following Python packages:
* os
* sys
* scipy
* datetime
* math
* numpy
* pandas
* uncertainties
* pyproj
* osgeo
* copy
* matplotlib

The run_glacier.py script reads the bed and surface topography in the data directory along the centerline in the centerlines directory. The topography is smoothed and plotted. Then, the Peclet number is calculated and plotted.

Everything needed to do the data processing is packaged within the glacier object (glacier.py)
