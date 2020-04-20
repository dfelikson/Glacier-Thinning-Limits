# Import modules
import os

from scipy.interpolate import interp1d
import scipy

import datetime

import math
import numpy as np
import pandas as ps

from uncertainties import unumpy as unp
from uncertainties import ufloat

from pyproj import Proj
from osgeo import gdal,ogr

import copy

import matplotlib.pyplot as plt

# DEBUG LEVEL
#  0 = no debug files
#  1 = ...
debug = 0


def uinterp1d(x, zu):
   z  = unp.nominal_values(zu)
   u  = unp.std_devs(zu)
   zvalid = z[~np.isnan(z)]
   uvalid = u[~np.isnan(z)]
   f = interp1d(x[~np.isnan(z)], 
               zvalid,
               bounds_error=False)
      
   zinterp = f(x)
   uinterp = uvalid[0] * np.ones(zinterp.shape)
   zuinterp = unp.uarray(zinterp, uinterp)
   
   return zuinterp   

import sys
sys.path.append("/home/student/denis/ScriptsAndUtilities/pythonModules")
from moving_average import *
from bilinear_interpolate import *

from Pe_moving_average import Pe_moving_average

# save mat files for debugging
import scipy.io as io


def moving_average_window_by_thickness(centerline_distance, bed_dataset, surface_dataset, thickness_factor):
   bed = bed_dataset.get_z()
   surface = surface_dataset.get_z()
   
   h = surface - bed
   window = np.nan * np.ones(h.shape)
   mvavgwindow = np.floor(thickness_factor * h)
   for idx in range(0,len(h),1):
      window[idx] = mvavgwindow[idx] / np.nanmean(np.diff(centerline_distance))   

      # Round n to odd number
      window[idx] = math.ceil(window[idx] / 2.) * 2 - 1

   valididx = ~np.isnan(window)
   xvalid = centerline_distance[valididx]
   windowvalid = window[valididx]
   f = interp1d(xvalid, windowvalid, kind='nearest', bounds_error=False)
   
   return f(centerline_distance)

def raster_along_centerline(rasterfilename, centerline, nodataValues, method='bilinear'):
   ds = gdal.Open(rasterfilename)
   rb = ds.GetRasterBand(1)
   gt = ds.GetGeoTransform()
   zarray = rb.ReadAsArray()

   data = np.empty(centerline.x.shape)
   zsample = np.empty(centerline.x.shape)

   imagexs = (centerline.x - gt[0]) / gt[1]
   imageys = (centerline.y - gt[3]) / gt[5]
   for i, imagex in enumerate(imagexs):
      imagey = imageys[i]

      if (imagex >= 0) and (imagey >= 0) and (imagex < ds.RasterXSize) and (imagey < ds.RasterYSize):

         if method == 'nearest':
            data[i] = zarray[int(np.floor(imagey)), int(np.floor(imagex))]
         elif method == 'bilinear':
            data[i] = bilinear_interpolate(zarray, imagex, imagey, nodataValues=nodataValues)
         else:
            data[i] = bilinear_interpolate(zarray, imagex, imagey, nodataValues=nodataValues)
         
      else:
         data[i] = np.nan

      if np.isnan(data[i]):
         zsample[i] = np.nan
      else:
         if np.any( np.abs(data[i] - nodataValues) < 1e-10 ):
            zsample[i] = np.nan
         else:
            zsample[i] = data[i]
   
   return zsample

def nearest_neighbor(dataset, centerline):
   ds = ogr.Open(dataset.filename)
   lyr = ds.GetLayer()

   xy  = np.array([0,0])
   xyz = np.array([0,0,0])
   for idx in range(lyr.GetFeatureCount()):
      feat = lyr.GetFeature(idx)
      geom = feat.GetGeometryRef()
      x = geom.GetX()
      y = geom.GetY()
      z = feat.GetField('chYtotal')

      # TBD: Hard-coded a reprojection from UTM 22N to Polar Stereographic
      p_utm = Proj(init="EPSG:32624",ellps='WGS84')
      p_ps  = Proj(proj='stere',lat_0=90,lat_ts=70,lon_0=-45,ellps='WGS84')
      lat, lon = p_utm(x, y, inverse=True)
      x_ps, y_ps = p_ps(lat, lon)

      xyz = np.vstack((xyz, [x_ps, y_ps, z]))
      xy  = np.vstack((xy,  [x_ps, y_ps]))

   xy  = xy[1:,:];
   xyz = xyz[1:,:];

   tree = scipy.spatial.cKDTree(xy)
   z = np.array([])
   d = np.array([])
   for i, x in enumerate(centerline.x):
      y = centerline.y[i]
      nearestneighbor = tree.query([x,y])

      z = np.append(z, xyz[nearestneighbor[1],2])
      d = np.append(d, nearestneighbor[0])

   z[d > 5000] = np.nan
   return z

def c_Dprime_D(h,dh,slope,dslope,flowLaw='hard-bed',**flow_parms):

   c = unp.uarray(np.nan * np.ones(slope.shape), np.nan * np.ones(slope.shape))
   D = unp.uarray(np.nan * np.ones(slope.shape), np.nan * np.ones(slope.shape))
   Dprime = unp.uarray(np.nan * np.ones(slope.shape), np.nan * np.ones(slope.shape))

   # Thresholds for all flow laws
   if 'slope_threshold' in flow_parms:
      slope_threshold = flow_parms['slope_threshold']
   else:
      slope_threshold = 0.01

   if flowLaw.find('hard-bed') >= 0:
      n = flow_parms['n']
      m = flow_parms['m']
      
		# Invalidate where slopes are too small
      nanidx = np.isnan(unp.nominal_values(slope))
      slope[nanidx] = ufloat(-9999.,np.nan)
      validIdx   = unp.nominal_values(slope) > slope_threshold
      invalidIdx = np.logical_not(validIdx)
      slope[nanidx] = ufloat(np.nan,np.nan)
      
      # Note: "Kb" has been factored out (see Cuffey and Paterson, 2010)
      c[validIdx] = (m+1) * (h[validIdx] * slope[validIdx])**m
      D[validIdx] = m * h[validIdx]**(m+1) * slope[validIdx]**(m-1)
      Dprime[validIdx] = m * (m+1) * h[validIdx]**m     * dh[validIdx]           * slope[validIdx]**(m-1) \
                         + m * (m-1) * h[validIdx]**(m+1) * slope[validIdx]**(m-2) * dslope[validIdx]

   if flowLaw.find('effective-pressure') >= 0:
      n = flow_parms['n']
      m = flow_parms['m']
      hw = flow_parms['hw']
      dhw = flow_parms['dhw']
      phatw = (1000./917.) * hw
      dphatw = (1000./917.) * dhw
      
      # Invalidate where ice is floating and where slopes are too small
      if 'flotation_threshold' in flow_parms:
         flotation_threshold = flow_parms['flotation_threshold']
      else:
         flotation_threshold = 75.0
      validIdx = np.logical_and(unp.nominal_values(h - phatw) > flotation_threshold, unp.nominal_values(slope) > slope_threshold)
      invalidIdx = np.logical_not(validIdx)

      # Note: "kappa" has been factored out (see Pfeffer, 2007)
      peff = h - phatw
      c[validIdx] = ( (h[validIdx]*slope[validIdx])**n/peff[validIdx]**m ) * ( 1 + ( (n-m)*h[validIdx] - n*phatw[validIdx] ) / peff[validIdx] )
      D[validIdx] = n * ( (slope[validIdx]**(n-1) * h[validIdx]**(n+1)) / peff[validIdx]**m )
      Dprime[validIdx] = ( n*(n-1)*slope[validIdx]**(n-2)*dslope[validIdx]*h[validIdx]**(n+1) ) / ( peff[validIdx]**m ) + \
                         ( n*(n+1)*slope[validIdx]**(n-1)*dh[validIdx]*h[validIdx]**n )         / ( peff[validIdx]**m ) - \
                         ( n*m*(dh[validIdx]-dphatw[validIdx])*slope[validIdx]**(n-1)*h[validIdx]**(n+1) ) / ( peff[validIdx]**(m+1) )

   return (c,Dprime,D)

def c_cPrime_D_Dprime(n,m,h,dh,slope,dslope):
   # Note: "Kb" has been factored out (see Cuffey and Paterson, 2010)
   c = (m+1) * (h * slope)**m
   cPrime = m * (m+1) * (h*slope)**(m-1) * (dh*slope + h*dslope)
   D = m * h**(m+1) * slope**(m-1)

   # Divide by zero check
   nonzeroidx = unp.nominal_values(slope)**(2-m) != 0
   zeroidx    = unp.nominal_values(slope)**(2-m) == 0
   
   Dprime = unp.uarray(np.nan * np.ones(c.shape), np.nan * np.ones(c.shape))
   Dprime[nonzeroidx] = m*(m+1)*h[nonzeroidx]**m * dh[nonzeroidx] * slope[nonzeroidx]**(m-1) \
                      + m*(m-1)*h[nonzeroidx]**(m+1) * slope[nonzeroidx]**(m-2) * dslope[nonzeroidx]
   
   D[zeroidx] = unp.uarray(np.nan, np.nan)
   c[zeroidx] = unp.uarray(np.nan, np.nan)

   return (c,cPrime,Dprime,D)

def Pe(glacier, centerline_distance, bed_dataset, surface_dataset, moving_average_thickness_factor, flowLaw='hard-bed', **flow_parms):
   # Parameters
   if 'n' in flow_parms:
      n = flow_parms['n']
   else:
      n = 3.
   if 'm' in flow_parms:
      m = flow_parms['m']
   else:
      m = 1.
   if 'moving_average_type' in flow_parms:
      moving_average_type = flow_parms['moving_average_type']
   else:
      moving_average_type = 'poly_fit'
   if 'slope_threshold' in flow_parms:
      slope_threshold = flow_parms['slope_threshold']
   else:
      slope_threshold = 0.01
   if 'flotation_threshold' in flow_parms:
      flotation_threshold = flow_parms['flotation_threshold']
   else:
      flotation_threshold = 75.0
   if 'discontinuity_cleanup' in flow_parms:
      discontinuity_cleanup = flow_parms['discontinuity_cleanup']
   else:
      discontinuity_cleanup = False
   if 'sea_level' in flow_parms:
      sea_level = flow_parms['sea_level']
   else:
      print('sea level not specified ... using 0 m')
      sea_level = 0.
   

   x = copy.deepcopy(centerline_distance)
   
   # Length of perturbation (l)
   l = copy.deepcopy(x)
   l[centerline_distance < surface_dataset.terminus] = np.nan  # nan out everything downglacier of terminus
   l = l - np.nanmin(l)                                        # shift l such that zero is at terminus

   errFlag = True
   bed = bed_dataset.get_zu()
   surface = surface_dataset.get_zu()
   if len(bed) == 0:
      errFlag = False
      bed = bed_dataset.get_z()
      surface = surface_dataset.get_z()

   # thickness
   h  = surface - bed
   # water height
   hw = sea_level - bed

   # Calculate moving average window
   moving_average_window = moving_average_window_by_thickness(centerline_distance,
                                                              bed_dataset,
                                                              surface_dataset,
                                                              moving_average_thickness_factor)

   # Moving average
   hufit, dhufit, slopeufit, dslopeufit, hwufit, dhwufit = Pe_moving_average(x, surface, h, hw, moving_average_window, fit_type=moving_average_type)
      
   # Set water depth to 0.0 where bed goes above sea level
   nanidx = np.isnan(unp.nominal_values(hwufit))
   hwufit [nanidx] = ufloat(-9999.,np.nan)
   dhwufit[nanidx] = ufloat(-9999.,np.nan)
   hwufit [np.where(unp.nominal_values(hwufit) < 0.0)] = 0.0
   dhwufit[np.where(unp.nominal_values(hwufit) < 0.0)] = 0.0
   hwufit [nanidx] = ufloat(np.nan,np.nan)
   dhwufit[nanidx] = ufloat(np.nan,np.nan)

   # Kinematic wave coefficients
   (c,Dprime,D) = c_Dprime_D(hufit, dhufit, slopeufit, dslopeufit, flowLaw=flowLaw, n=n, m=m, hw=hwufit, dhw=dhwufit, slope_threshold=slope_threshold, flotation_threshold=flotation_threshold)

   # Peclet number
   Peclet = ( (c - Dprime) / D ) * l

   # Remove values before/after discontinuities until slope break
   if discontinuity_cleanup:
      nan_idxs = np.where(~np.isnan(unp.nominal_values(Peclet)))[0]
      if len(nan_idxs) > 0:
         nan_idx_diffs = np.diff(nan_idxs)
         nan_idx_diffs = np.insert(nan_idx_diffs, 0, 0)
         nan_idxs = nan_idxs[nan_idx_diffs > 1]
         nan_idx_diffs = nan_idx_diffs[nan_idx_diffs > 1]
         if len(nan_idxs) > 0:
            slopes = np.diff(unp.nominal_values(Peclet))
            slopesigns = np.sign(slopes)
            dslopesigns = np.diff(slopesigns)
            for i, idx in enumerate(nan_idxs):
               start_idx = idx
               end_idx   = idx + nan_idx_diffs[i]
               # Find nearest switch in slope sign
               delete_start = np.where(dslopesigns[0:start_idx] > 0.0)[0]
               if len(delete_start) == 0:
                  Peclet[ 0.0 : start_idx+1]  = np.nan
               else:
                  Peclet[ np.max( np.where(dslopesigns[0:start_idx] > 0.0) )+2 : start_idx+1]  = np.nan
               delete_end = np.where(dslopesigns[end_idx:] > 0.0)[0]
               #if len(delete_end) == 0:
               #   Peclet[ end_idx : end_idx + np.min( np.where(dslopesigns[end_idx:] > 0.0) )] = np.nan
               invalid_idx = np.where(dslopesigns[end_idx:] > 0.0)[0]
               if len(invalid_idx) > 0:
                  Peclet[ end_idx : end_idx + np.min( np.where(dslopesigns[end_idx:] > 0.0) )] = np.nan
      
   return (c, Dprime, D, Peclet)

###########################################################################
class glacier:
   def __init__(self, centerline_filename=None, centerline_sample_distance=50., reverseOpt=False, interpDistThreshold=None, xcol=0.0, ycol=1.0):
      self.centerline = glacier.centerline()
      self.centerline.shapefile = centerline_filename
      self.centerline.sample_distance = centerline_sample_distance
      if centerline_filename:
         self.read_centerline(reverseOpt, interpDistThreshold, xcol, ycol)
      self.datasets = {}

   name = ''
   abbr = ''

   # Centerline
   class centerline:
      shapefile = ''
      x = np.array([])
      y = np.array([])
      data = np.array([])
      distance = np.array([])
      sample_distance = None

   def read_centerline(self, reverseOpt, interpDistThreshold, xcol, ycol):
      
      if self.centerline.shapefile.endswith('.shp'):
         ds = ogr.Open(self.centerline.shapefile)
         lyr = ds.GetLayer()
         feat = lyr.GetFeature(0)
         geom = feat.GetGeometryRef()
         
         x = np.array([])
         y = np.array([])
         for i, pointxy in enumerate(geom.GetPoints()):
            xtmp, ytmp = pointxy
            x  = np.append(x,  xtmp)
            y  = np.append(y,  ytmp)
      
      elif self.centerline.shapefile.endswith('.csv'):
         data = np.genfromtxt(self.centerline.shapefile, delimiter=',')
         data = data[~np.isnan(data).any(axis=1)]
         x = data[:,xcol]
         y = data[:,ycol]
        
      else:
         print("ERROR (glacier.py): invalid centerline filename given")
         return

      # Reverse the centerline
      if reverseOpt:
         x = x[::-1]
         y = y[::-1]
         if self.centerline.shapefile.endswith('.csv'):
            data = np.flipud(data)

      for i, val in enumerate(x):
         if i > 0:
            d = np.append(d, np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ))
         else:
            d = np.array(0)

      d = np.cumsum(d)

      # Resample
      sample_distance = self.centerline.sample_distance
      dnew = np.arange(0, d[-1], sample_distance)
      distances = np.array([])
      if interpDistThreshold is not None:
         for dnewi in dnew:
            distances = np.append(distances, np.min( np.abs(dnewi - d) ))
         dnew[distances > interpDistThreshold] = np.nan
         #dnew = np.where(distances > interpDistThreshold, np.nan, distances)
      
      f = interp1d(d, x)
      xnew = f(dnew)
      f = interp1d(d, y)
      ynew = f(dnew)

      if self.centerline.shapefile.endswith('.csv'):
         datanew = np.empty( (len(dnew), data.shape[1]) )
         for i in range(0,data.shape[1]):
            z = data[:,i]
            f = interp1d(d, z)
            znew = f(dnew)
            datanew[:,i] = znew
            self.centerline.data = datanew

      self.centerline.distance = dnew
      self.centerline.x = xnew
      self.centerline.y = ynew

   # Datasets
   datasets = {}
   class dataset:
      datetime = datetime.datetime(1900, 1, 1, 00, 00, 00)
      type = ''
      filename = ''
      nodataValues = np.nan

      centerlineDistance = np.array([])
      z = np.array([])
      zu = np.array([])
      moving_average_thickness_factor = None
      zmvavg = np.array([])
      zumvavg = np.array([])
      valuecol = None

      extrapolatedIdx = np.array([])
      validIdx = np.array([])

      uncertainty = None

      terminus = np.nan

      method = 'bilinear'

      def get_z(self, centerlineDistance=None):
         x = copy.deepcopy(self.centerlineDistance)
         z = copy.deepcopy(self.z)
         z[~self.validIdx] = np.nan

         if centerlineDistance is not None:
            idx = (np.abs(x-centerlineDistance)).argmin()
            return z[idx]
         else:
            return z

      def get_zmvavg(self, centerlineDistance=-1):
         x = copy.deepcopy(self.centerlineDistance)
         zmvavg = copy.deepcopy(self.zmvavg)
         if len(zmvavg) == 0:
            return []
         zmvavg[~self.validIdx] = np.nan

         if centerlineDistance >= 0:
            f = interp1d(x[~np.isnan(zmvavg)], 
                         zmvavg[~np.isnan(zmvavg)],
                         bounds_error=False)
                         
            return f(centerlineDistance)
   
         else:
            return zmvavg

      def get_zu(self, centerlineDistance=-1):
         x  = copy.deepcopy(self.centerlineDistance)
         zu = copy.deepcopy(self.zu)
         if len(zu) > 0:
            if len(zu[~self.validIdx]) > 0:
               zu[~self.validIdx] = unp.uarray(np.nan, np.nan)

            if centerlineDistance > 0:
               idx = (np.abs(x-centerlineDistance)).argmin()
               return zu[idx]
            else:
               return zu
         else:
            return zu

      def get_zumvavg(self, centerlineDistance=-1):
         x  = copy.deepcopy(self.centerlineDistance)
         zumvavg = copy.deepcopy(self.zumvavg)
         zumvavg[~self.validIdx] = unp.uarray(np.nan, np.nan)

         if centerlineDistance > 0:
            idx = (np.abs(x-centerlineDistance)).argmin()
            return zumvavg[idx]
         else:
            return zumvavg
      
      def set_zu(self, zumvavg):
         self.zu = zumvavg
         self.z  = unp.nominal_values(zumvavg)

      def set_zumvavg(self, zumvavg):
         self.zumvavg = zumvavg
         self.zmvavg  = unp.nominal_values(zumvavg)

      def fill_z(self):
         validIdx = ~np.isnan(self.z)

         f = interp1d(self.centerlineDistance[validIdx], 
                      self.z[validIdx],
                      bounds_error=False, fill_value=np.nan)
                         
         zfill = f(self.centerlineDistance[~validIdx])
         self.z[~validIdx] = zfill
         self.validIdx[~validIdx] = bool(1)

      def set_terminus(self, terminus):
         self.terminus = terminus
         if not np.isnan(terminus):
            self.validIdx[self.centerlineDistance < terminus] = False


   moving_average_thickness_factor = np.nan
   moving_average_window = np.nan
   def moving_average(self, bed, surface, average):
      # Calculate window
      if self.datasets[average].moving_average_thickness_factor is not None:
         thickness_factor = self.datasets[average].moving_average_thickness_factor
      else:
         window = moving_average_window_by_thickness(self.centerline.distance, 
                                                     self.datasets[bed], 
                                                     self.datasets[surface], 
                                                     self.moving_average_thickness_factor)
      
      self.moving_average_window = window

      # Interpolate and moving average zu
      if self.datasets[average].uncertainty:
         zu = self.datasets[average].get_zu()
         zuinterp = uinterp1d(self.centerline.distance, zu)
         zumvavg = umoving_fit(self.centerline.distance, zuinterp, window)
         self.datasets[average].zumvavg = zumvavg

      # Interpolate and moving average z
      z = self.datasets[average].get_z()
      zvalid = z[~np.isnan(z)]
      x = self.centerline.distance[~np.isnan(z)]
      f = interp1d(x, zvalid, bounds_error=False)
      zinterp = f(self.centerline.distance)
      zmvavg  = moving_fit(self.centerline.distance, zinterp,  window)
      self.datasets[average].zmvavg  = zmvavg
      
   def mask(self, datasetToMask, mask, validValue):
      if type(validValue) is float:
         maskVals = self.datasets[mask].get_z()
         maskIdx  = maskVals == validValue
      if type(validValue) is str:
         maskVals = self.datasets[mask].get_z()
         maskIdx  = eval('maskVals' + validValue)
         
      self.datasets[datasetToMask].z[ ~maskIdx.astype(bool)] = np.nan
      self.datasets[datasetToMask].zu[~maskIdx.astype(bool)] = unp.uarray(np.nan, np.nan)
      if self.datasets[datasetToMask].zmvavg is not None and len(self.datasets[datasetToMask].zmvavg) > 0:
         self.datasets[datasetToMask].zmvavg[ ~maskIdx.astype(bool)] = np.nan
         self.datasets[datasetToMask].zumvavg[~maskIdx.astype(bool)] = unp.uarray(np.nan, np.nan)

      

   def new_dataset(self, name, **kwargs):
      self.datasets[name] = copy.deepcopy(self.dataset())
      self.datasets[name].centerlineDistance = copy.deepcopy(self.centerline.distance)
      for key in ('type', 'filename', 'nodataValues', 'terminus', 'method', 'datetime', 'valuecol'):
         if key in kwargs:
            setattr(self.datasets[name], key, kwargs[key])

      if self.datasets[name].type == 'raster':
         self.datasets[name].z = raster_along_centerline(self.datasets[name].filename, self.centerline, self.datasets[name].nodataValues, self.datasets[name].method)
      elif self.datasets[name].type == 'pointcloud':
         self.datasets[name].z = nearest_neighbor(self.datasets[name].filename, self.centerline, self.datasets[name].nodataValues)
      elif self.datasets[name].type == 'centerline':
         self.datasets[name].z = self.centerline.data[:,self.datasets[name].valuecol]

      # Uncertainty
      if 'uncertainty' in kwargs:
         self.datasets[name].uncertainty = kwargs['uncertainty']
         if isinstance(kwargs['uncertainty'], str):
            # Read uncertainty from raster
            u = raster_along_centerline(self.datasets[name].uncertainty, self.centerline, self.datasets[name].nodataValues)
         if isinstance(kwargs['uncertainty'], int):
            # Convert to float
            u = float(self.datasets[name].uncertainty)
         if isinstance(kwargs['uncertainty'], float):
            # Store the constant value
            u = kwargs['uncertainty']
         
         u = np.where(u<0, np.nan, u)
         self.datasets[name].zu = unp.uarray(self.datasets[name].z, u)

      # Set up validIdx and ExtrapolatedIdx
      self.datasets[name].extrapolatedIdx = np.zeros(self.centerline.distance.shape, dtype=bool)

      if not np.isnan(self.datasets[name].terminus):
         self.datasets[name].validIdx = self.centerline.distance >= self.datasets[name].terminus
      else:
         self.datasets[name].validIdx = self.centerline.distance == self.centerline.distance

      if 'invalidCenterlineDistances' in kwargs:
         if np.any(kwargs['invalidCenterlineDistances']):
            for invalidXY in kwargs['invalidCenterlineDistances']:
               self.datasets[name].validIdx[np.where( (self.centerline.distance >= invalidXY[0]) & 
                                                      (self.centerline.distance <= invalidXY[1]) )] = bool(0)

      self.datasets[name].validIdx[np.isnan(self.datasets[name].z)] = bool(0)


      # Extrapolate the start of the data by just grabbing the first valid point (e.g. the nearest neighbor)
      if 'extrapolateStart' in kwargs:
         validIdx = np.where(self.datasets[name].validIdx == 1)
         if validIdx:
            firstValidIdx = np.min(validIdx)
            firstValidValue = self.datasets[name].z[firstValidIdx]
            self.datasets[name].z[0:firstValidIdx] = firstValidValue
            self.datasets[name].validIdx[0:firstValidIdx] = bool(1)
            self.datasets[name].extrapolatedIdx[0:firstValidIdx] = bool(1)
            if 'uncertainty' in kwargs:
               firstValidUncert = unp.std_devs(self.datasets[name].zu[firstValidIdx])
               self.datasets[name].zu[0:firstValidIdx] = unp.uarray(firstValidValue, firstValidUncert)


   def adjacent_centerline(self, distance):
      adjacent_centerline(self.centerline, distance)

   # Peclet number
   #@profile
   def Pe(self, bed, surface, name, **kwargs):
      if self.datasets[name].moving_average_thickness_factor is not None:
         moving_average_thickness_factor = self.datasets[name].moving_average_thickness_factor
      else:
         moving_average_thickness_factor = self.moving_average_thickness_factor
      (c, Dprime, D, Peclet) = Pe(self.abbr, self.centerline.distance, self.datasets[bed], self.datasets[surface], moving_average_thickness_factor, **kwargs)
      self.datasets[name].zumvavg = Peclet
      self.datasets[name].zmvavg = unp.nominal_values(Peclet)
      return(c, Dprime, D)

   # Surface elevation change
   def dh(self, surface1, surface2):
      return self.datasets[surface2].zmvavg - self.datasets[surface1].zmvavg

