#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:58:27 2026

@author: Jon Paul Lundquist
"""

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u

# --- Scan parameters and grid ---

def _build_scan_params():
    minN = 6 # Minimum number of events in a wedge
    # Lower bound energy cut: maximizes tau supergalactic curvature for Auger Open Data. 
    min_Ecut = 13    # 10 EeV in original TA paper.
 
    # Original Paper Scan parameters
    # distances = np.arange(15, 95, 5)[::-1]
    # widths = np.arange(10, 95, 5)[::-1] / 2 
    # directions = np.arange(0,360,5)
    # Ecuts = np.linspace(10, 80, 15)
    
    # A finer scan allowed by significantly improved runtime.
    distances = np.arange(14, 92, 2)[::-1]
    widths = np.arange(4, 92, 2)[::-1] / 2
    directions = np.arange(0,360,2)
    Ecuts = np.arange(10,75,5) # 75/80 EeV never results in maximum significant scan.
    
    # Many NumPy operations (and lots of C/Numba code) are fastest on contiguous arrays.
    distances = distances.astype(np.float32, order='C')
    widths = widths.astype(np.float32, order='C')
    directions = directions.astype(np.float32, order='C')
    Ecuts = Ecuts.astype(np.float32, order='C')

    return minN, min_Ecut, distances, widths, directions, Ecuts


def _build_scan_grid(grid_separation=1, max_dec=25):
    # Create supergalactic grid of scan centers and apply Auger field-of-view limit.
    # Original paper has an approximately equal separation grid of 2 degrees.
    grid, ipix, nside = _grid_equal(grid_separation) 
    mask = grid.dec.deg <= max_dec # Auger exposure maximum is +25 deg. declination
    grid = grid[mask]
    ipix = ipix[mask]

    return grid.transform_to('supergalactic'), ipix, nside


def _grid_equal(separation):
    # Create a Healpix grid of (approximately) equal separation on the sky
    # This is the grid of centers for the wedge correlation scan

    # List of reasonable Nside values for the grid
    Nside = np.array((2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9))
    # Convert the Nside values to center separations in degrees
    seps = hp.nside2resol(Nside, arcmin=True) / 60
    # Find the Nside value that is closest to the desired separation
    ind = np.argmin(abs(seps - separation)) 
    nside = int(Nside[ind])

    npix = hp.nside2npix(nside) # Calculate the number of pixels in the grid
    ipix = np.arange(npix) # Create an array of pixel indices

    # Convert the pixel indices to theta and phi
    theta, phi = hp.pix2ang(nside, ipix, nest=False) 
    # Convert the Healpix theta (0 to pi) to declination (-90 to 90 degrees)
    dec = np.degrees(0.5*np.pi - theta) 
    ra  = np.degrees(phi) # phi is equivalent to right ascension

    # Create a True Equator, True Equinox SkyCoord object (system used for Auger data) 
    # with the right ascension and declination
    grid = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='tete') 
    return grid, ipix, nside