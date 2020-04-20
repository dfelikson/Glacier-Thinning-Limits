#!/usr/bin/env python

import sys
import numpy as np
from uncertainties import unumpy as unp

#from statsmodels.nonparametric.smoothers_lowess import lowess
from polynomial_fits import *

def Pe_moving_average(x, surface, thickness, water_depth, moving_average_window, fit_type='poly_fit'):

   # Initialize
   # {{{
   ns           = np.nan * np.ones(len(x))
   nh           = np.nan * np.ones(len(x))
   sfit         = np.nan * np.ones(len(x))
   sfiterr      = np.nan * np.ones(len(x))
   hfit         = np.nan * np.ones(len(x))
   hfiterr      = np.nan * np.ones(len(x))
   hwfit        = np.nan * np.ones(len(x))
   hwfiterr     = np.nan * np.ones(len(x))
   dhfit        = np.nan * np.ones(len(x))
   dhfiterr     = np.nan * np.ones(len(x))
   dhwfit       = np.nan * np.ones(len(x))
   dhwfiterr    = np.nan * np.ones(len(x))
   slopefit     = np.nan * np.ones(len(x))
   slopefiterr  = np.nan * np.ones(len(x))
   dslopefit    = np.nan * np.ones(len(x))
   dslopefiterr = np.nan * np.ones(len(x))

   sufit      = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   hufit      = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   dhufit     = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   slopeufit  = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   dslopeufit = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   hwufit     = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   dhwufit    = unp.uarray(np.nan * np.ones(len(x)), np.nan * np.ones(len(x)))
   #}}}

   s     = unp.nominal_values(surface)
   serr  = unp.std_devs(surface)
   h     = unp.nominal_values(thickness)
   herr  = unp.std_devs(thickness)
   hw    = unp.nominal_values(water_depth)
   hwerr = unp.std_devs(water_depth)

   # OPTION fit polynomials but NOT at the end points, where window extends beyond valid data
   # This is also called a savitzky_golay filter (there's a python function to do this)
   if fit_type == 'poly_fit': #{{{
      firstValid = np.min(np.where(~np.isnan(unp.nominal_values(surface))))
      lastValid  = np.max(np.where(~np.isnan(unp.nominal_values(surface))))
      
      for idx in range(firstValid, lastValid, 1):
         # Find elements within window
         idx_start = np.max( (0     , np.floor(idx - moving_average_window[idx]/2)) )
         idx_stop  = np.min( (len(h), np.floor(idx + moving_average_window[idx]/2)) )

         if ~np.isnan(idx_start) and ~np.isnan(idx_stop):
            idx_start = int(idx_start)
            idx_stop  = int(idx_stop)
            valididx  = ~np.isnan(s[idx_start:idx_stop])
            
            # Find valid entries within the window
            xwindowvalid     = x[idx_start:idx_stop][valididx]
            swindowvalid     = s[idx_start:idx_stop][valididx]
            serrwindowvalid  = serr[idx_start:idx_stop][valididx]
            hwindowvalid     = h[idx_start:idx_stop][valididx]
            herrwindowvalid  = herr[idx_start:idx_stop][valididx]
            hwwindowvalid    = hw[idx_start:idx_stop][valididx]
            hwerrwindowvalid = hwerr[idx_start:idx_stop][valididx]

            if idx_start >= firstValid and idx_stop <= lastValid:
               sfit[idx], sfiterr[idx], slopefit[idx], slopefiterr[idx], dslopefit[idx], dslopefiterr[idx]    = fit_second_order(xwindowvalid, swindowvalid, serrwindowvalid, xfit=x[idx])
               hfit[idx],  hfiterr[idx],  dhfit[idx],     dhfiterr[idx]                                       = fit_first_order( xwindowvalid, hwindowvalid, herrwindowvalid, xfit=x[idx])
               hwfit[idx], hwfiterr[idx], dhwfit[idx],    dhwfiterr[idx]                                      = fit_first_order( xwindowvalid, hwwindowvalid, hwerrwindowvalid, xfit=x[idx])
      
      invalididx = np.isnan(unp.nominal_values(surface))
      sfit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);      sfiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      slopefit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);  slopefiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      dslopefit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape); dslopefiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      hfit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);      hfiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      dhfit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);     dhfiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      hwfit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);      hwfiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      dhwfit[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape);     dhwfiterr[invalididx] = np.nan * np.ones(np.where(invalididx)[0].shape)
      
      hufit      = unp.uarray(hfit,      hfiterr)
      dhufit     = unp.uarray(dhfit,     dhfiterr)
      slopeufit  = unp.uarray(slopefit,  slopefiterr)
      dslopeufit = unp.uarray(dslopefit, dslopefiterr)
      hwufit     = unp.uarray(hwfit,     hwfiterr)
      dhwufit    = unp.uarray(dhwfit,    dhwfiterr)
   #}}}

   return hufit, dhufit, slopeufit, dslopeufit, hwufit, dhwufit


