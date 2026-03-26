#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:33:09 2023

@author: Jon Paul Lundquist
"""

import numpy as np
import numpy.random as nr
import utm
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord


def _iso_MC_Auger(data, E_cut=None, N_mc=None, save=False, seed=None):
    if seed is not None:
        rng = nr.default_rng(seed)

    else:
        rng = nr.default_rng()

    # Lowest energy cut
    if E_cut is None:
        Ecut = float(np.min(data["sd_energy"]))
    else:
        Ecut = float(E_cut)

    # Select events with energy >= Ecut
    data = data[data['sd_energy']>=Ecut]
    
    # Number of Monte Carlo events in dataset
    if N_mc == None:
        N_mc = data.size
        
    d_energy = data['sd_energy'] # Surface detector energy in EeV
    # Eastward-,northward-coordinate and altitude of shower core (UTM coordinate system)
    d_easting = data['sd_easting']
    d_northing = data['sd_northing']
    d_altitude = data['sd_altitude']
    # Horizontal coordinates of shower core
    d_zenith = data['sd_theta']
    d_azimuth = data['sd_phi']
    # Time of shower core arrival at the surface detector
    d_time = data['gpstime'] + data['sd_gpsnanotime']*10**-9
    
    # Scramble in blocks of about 4.5 months to preserve observed energy-time corr.
    idx_arr = _permutation_within_time_blocks(d_time, block_size=25, rng=rng)
    
    # Detector state fixed. Preserve time-core coordinate correlations.
    mc_time     = d_time
    mc_easting  = d_easting
    mc_northing = d_northing
    mc_altitude = d_altitude

    # Keep detector measurement correlations intact (energy-zenith, zenith-azimuth). 
    # Scramble in blocks to preserve observed energy-time correlation.
    mc_zenith   = d_zenith[idx_arr]
    mc_azimuth  = d_azimuth[idx_arr]
    mc_energy   = d_energy[idx_arr]
    # Convert local coordinates to Equatorial True Equator, True Equinox coordinates
    mc_coords = _local_to_equatorial(mc_easting, mc_northing, mc_altitude, mc_time, 
                                     mc_zenith, mc_azimuth)

    # Right Ascension and Declination of apparent source
    ra_deg  = mc_coords.ra.deg
    dec_deg = mc_coords.dec.deg

    dtype = [("time", "f8"), ("easting", "f8"), ("northing", "f8"), ("altitude", "f8"),
             ("zenith", "f8"), ("azimuth", "f8"), ("energy", "f8"),("ra", "f8"),
             ("dec", "f8")]
    
    mc_data = np.empty(N_mc, dtype=dtype)
    mc_data["time"]     = mc_time
    mc_data["easting"]  = mc_easting
    mc_data["northing"] = mc_northing
    mc_data["altitude"] = mc_altitude
    mc_data["zenith"]   = mc_zenith
    mc_data["azimuth"]  = mc_azimuth
    mc_data["energy"]   = mc_energy
    mc_data["ra"]       = ra_deg
    mc_data["dec"]      = dec_deg

    if save:
        outpath = save if isinstance(save, (str, bytes)) else "iso_MC_Auger.npz"
        np.savez_compressed(
            outpath,
            ra=ra_deg.astype(np.float64, copy=False),
            dec=dec_deg.astype(np.float64, copy=False),
            energy=mc_energy.astype(np.float64, copy=False),
            mc_data=mc_data,
            Ecut=np.array(Ecut, dtype=np.float64),
        )
    
    return mc_coords, mc_energy, mc_data


def _load_iso_MC_Auger_npz(path):
    """
    Load isotropic MC saved by `iso_MC_Auger(..., save=...)`.

    Returns:
      mc_coords (SkyCoord in 'tete'), mc_energy (ndarray), mc_data (structured ndarray)
    """
    d = np.load(path)
    ra = d["ra"]
    dec = d["dec"]
    mc_energy = d["energy"]
    mc_data = d["mc_data"]
    mc_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="tete")
    return mc_coords, mc_energy, mc_data

# Scramble in blocks to preserve observed energy-time correlation.
def _permutation_within_time_blocks(d_time, block_size, rng):
    n = d_time.size
    order = np.argsort(d_time)
    idx_arr = np.arange(n)

    for start in range(0, n, block_size):
        block = order[start:start + block_size]
        if block.size > 1:
            idx_arr[block] = block[rng.permutation(block.size)]
    return idx_arr

# Convert local coordinates to Equatorial True Equator, True Equinox coordinates
def _local_to_equatorial(easting, northing, altitude, time, zenith, azimuth):
    latitude, longitude = utm.to_latlon(easting.copy(), northing.copy(), 19, 
                                        northern=False) #19 is the UTM zone for Auger
    # Create EarthLocation object for Auger
    location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg, 
                             height=altitude * u.m)
    obs_time = Time(time, format='gps')
    # Create AltAz object for Auger
    #Auger azimuth is 0 deg East counter-clockwise
    altaz = AltAz(alt=(90 - zenith) * u.deg, az=(360 - azimuth + 90) % 360 * u.deg, 
                  location=location, obstime=obs_time)

    # Convert Horizontal coordinates to Equatorial True Equator, True Equinox coords.
    coords = SkyCoord(altaz).transform_to('tete')
 
    return coords

