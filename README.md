# Glacier-Thinning-Limits
The Glacier Thinning Limits tool can be used to calculate the Peclet Number along glacier flowlines, that describes how terminus-initiated thinning will evolve. The code here accompanies [Felikson et al. (2018)](https://doi.org/10.1038/ngeo2934).

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

Everything needed to do the data processing is packaged within the glacier object (glacier.py)
