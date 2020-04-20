#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

# Fit first order polynomial to water depth
def fit_first_order(x, y, yerr, xfit=None):
#{{{

   yfit=None
   yfiterr=None
   dyfit=None
   dyfiterr=None

   # Shift coordinates
   xshift = np.mean(x)
   x = x - xshift
   yshift = np.mean(y)
   y = y - yshift

   if len(y) > 3:
      n = len(y)
      H = np.matrix(np.hstack( (x[np.newaxis].T, np.ones(n)[np.newaxis].T) ))
      W = np.matrix(np.diag(1. / yerr**2 ))
      V = np.linalg.inv( np.transpose(H) * W * H )
      p = V * np.transpose(H) * W * y[np.newaxis].T
      
      # If x-coordinate not provided, perform the fit at all given x's
      # ... otherwise, evaluate the polynomial fit at just that coordinate
      if xfit is None:
         xfit = x
      else:
         xfit = xfit - xshift

      yfit      = np.polyval(p, xfit) + yshift
      yfiterr   = np.sqrt(V[1,1]) # TBD: This should probably be the RSS of the post-fit residuals
      hwfiterr  = np.sqrt(V[1,1]) # TBD: This should probably be the RSS of the post-fit residuals
      dyfit     = p[0]
      dyfiterr  = np.sqrt(V[0,0]) # TBD: What should this be?

   return yfit, yfiterr, dyfit, dyfiterr
#}}}

# Fit second order polynomial to surface
def fit_second_order(x, y, yerr, xfit=None):
#{{{
   yfit=None
   yfiterr=None
   dyfit=None
   dyfiterr=None
   ddyfit=None
   ddyfiterr=None

   # Shift coordinates
   xshift = np.mean(x)
   x = x - xshift
   yshift = np.mean(y)
   y = y - yshift

   if len(y) > 4:
      n = len(y)
      H = np.matrix(np.hstack( (x[np.newaxis].T**2, x[np.newaxis].T, np.ones(n)[np.newaxis].T) ))
      W = np.matrix(np.diag(1. / yerr**2 ))
      V = np.linalg.inv( np.transpose(H) * W * H )
      p = V * np.transpose(H) * W * y[np.newaxis].T

      # If x-coordinate not provided, perform the fit at all given x's
      # ... otherwise, evaluate the polynomial fit at just that coordinate
      if xfit is None:
         xfit = x
      else:
         xfit = xfit - xshift

      yfit         = np.polyval(p, xfit) + yshift
      yfiterr      = np.sqrt(V[2,2])      # TBD: Placeholder - this needs to be propagated through second order polynomial
                                          # => Var[sfit] = [1 x x^2] * V * [1 x x^2]'
      dyfit        = np.polyval(p[0:2], xfit)
      dyfiterr     = np.sqrt(V[1,1]) # TBD: Placeholder - this needs to be propagated through second order polynomial
      ddyfit       = p[0]
      ddyfiterr    = np.sqrt(V[0,0]) # TBD: Placeholder - this needs to be propagated through second order polynomial

   return yfit, yfiterr, dyfit, dyfiterr, ddyfit, ddyfiterr
#}}}

