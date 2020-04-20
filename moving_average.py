#!/bin/env python

import numpy as np
from uncertainties import unumpy as unp

from matplotlib import pyplot as plt


def umoving_fit(x, yu, window):
#{{{
   ymvavg = np.nan * np.ones(yu.shape)
   umvavg = np.nan * np.ones(yu.shape)
   
   for idx in xrange(0,len(yu),1):
      # Find elements within window
      if type(window).__module__ == np.__name__:
         if np.isnan(window[idx]):
            continue
         idx_start  = int(np.max( (0      , np.floor(idx - window[idx]/2)) ))
         idx_stop   = int(np.min( (len(yu), np.floor(idx + window[idx]/2)) ))
      else:
         idx_start  = int(np.max( (0      , np.floor(idx - window/2)) ))
         idx_stop   = int(np.min( (len(yu), np.floor(idx + window/2)) ))
         if np.isnan(window):
            continue

      if ~np.isnan(idx_start) and ~np.isnan(idx_stop):
         xwindow    = x[idx_start:idx_stop]
         yuwindow   = yu[idx_start:idx_stop]
         ywindow    = unp.nominal_values(yuwindow)
         uwindow    = unp.std_devs(yuwindow)

         # Find non-nan entries
         valididx = ~np.isnan(unp.nominal_values(yuwindow))
         xvalid = xwindow[valididx]
         yvalid = ywindow[valididx]
         uvalid = uwindow[valididx]

         n = len(xvalid)
         if n > 2:
            # Regress
            H = np.matrix(np.hstack( (xvalid[np.newaxis].T, np.ones(n)[np.newaxis].T) ))
            if np.linalg.norm(uvalid) != 0:
               W = np.matrix(np.diag(1. / uvalid**2 ))
            else:
               W = np.identity(len(uvalid))
            V = np.linalg.inv( np.transpose(H) * W * H )
            p = V * np.transpose(H) * W * yvalid[np.newaxis].T

            ymvavg[idx] = np.polyval(p, x[idx])
            prediction_var = np.matrix(np.array([x[idx], 1])) * V * np.matrix(np.array([x[idx], 1])).T
            umvavg[idx] = np.sqrt(prediction_var)
            
   ymvavg[np.isnan(unp.nominal_values(yu))] = np.nan
   umvavg[np.isnan(unp.nominal_values(yu))] = np.nan
   yumvavg = unp.uarray(ymvavg, umvavg)
   return yumvavg
#}}}

def moving_fit(x, y, window):
#{{{
   ymvavg = np.array(np.nan * np.ones(y.shape))

   for idx in range(0,len(y),1):
      # Find elements within window
      if type(window).__module__ == np.__name__:
         if np.isnan(window[idx]):
            continue
         idx_start  = int(np.max( (0     , np.floor(idx - window[idx]/2)) ))
         idx_stop   = int(np.min( (len(y), np.floor(idx + window[idx]/2)) ))
      else:
         if np.isnan(window):
            continue
         idx_start  = int(np.max( (0     , np.floor(idx - window/2)) ))
         idx_stop   = int(np.min( (len(y), np.floor(idx + window/2)) ))

      if ~np.isnan(idx_start) and ~np.isnan(idx_stop):
         xwindow    = x[idx_start:idx_stop]
         ywindow    = y[idx_start:idx_stop]

         # Find non-nan entries
         valididx = ~np.isnan(ywindow)
         xvalid = xwindow[valididx]
         yvalid = ywindow[valididx]

         n = len(xvalid)
         if n > 2:
            # Regress
            H = np.matrix(np.hstack( (xvalid[np.newaxis].T, np.ones(n)[np.newaxis].T) ))
            #W = np.matrix(np.diag(1. / 1.**2 ))
            V = np.linalg.inv( np.transpose(H) * H ) # np.linalg.inv( np.transpose(H) * W * H )
            p = V * np.transpose(H) * yvalid[np.newaxis].T # V * np.transpose(H) * W * yvalid[np.newaxis].T

            ymvavg[idx] = np.polyval(p, x[idx])

   ymvavg[np.isnan(y)] = np.nan
   return ymvavg
#}}}

def moving_average_downsample(x, y, window):
   window = int(window)
   cumsum = np.cumsum(np.insert(y, 0, 0))
   y_mvavg = (cumsum[window:] - cumsum[:-window]) / float(window)
   y_mvavg = y_mvavg[::window]
   return y_mvavg

#def umoving_average(x, yu, window):
#   yumvavg = unp.uarray(np.nan * np.ones(yu.shape), np.nan * np.ones(yu.shape))
#
#   for idx, element in enumerate(yu):
#      # Find elements within window
#      idx_start  = np.max( (0      , np.floor(idx - window/2)) )
#      idx_stop   = np.min( (len(yu), np.floor(idx + window/2)) )
#      yuwindow   = yu[idx_start:idx_stop]
#
#      # Find non-nan entries
#      valididx = ~np.isnan(unp.nominal_values(yuwindow))
#      
#      # Average over window
#      if np.any(valididx):
#         yumvavg[idx] = yuwindow[valididx].mean()
#      else:
#         yumvavg[idx] = np.nan
#      
#      # Insert nans into smoothed array where original array had nans
#      yumvavg[np.isnan(unp.nominal_values(yu))] = unp.uarray(np.nan, np.nan)
#      
#   return yumvavg

def moving_average(x, y, window):
#{{{
   ymvavg  = np.array(np.nan * np.ones(y.shape))
   ystddev = np.array(np.nan * np.ones(y.shape))

   for idx, element in enumerate(y):
      if np.isnan(window):
         continue

      # Find elements within window
      idx_start = int(np.max( (0     , np.floor(idx - window/2)) ))
      idx_stop  = int(np.min( (len(y), np.floor(idx + window/2)) ))
      ywindow   = y[idx_start:idx_stop]

      # Find non-nan entries
      valididx = ~np.isnan(ywindow)
      
      # Average over window
      if np.any(valididx):
         ymvavg[idx]  = ywindow[valididx].mean()
         ystddev[idx] = ywindow[valididx].std()
      else:
         ymvavg[idx]  = np.nan
         ystddev[idx] = np.nan

      # Insert nans into smoothed array where original array had nans
      ymvavg[np.isnan(y)]  = np.nan
      ystddev[np.isnan(y)] = np.nan
      
   return ymvavg, ystddev
#}}}

