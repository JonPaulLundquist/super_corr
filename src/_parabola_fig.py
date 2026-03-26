#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:59:51 2026

@author: Jon Paul Lundquist
"""
    
import numpy as np
from _parabola_fit import _parabola, _parabola_fit
import matplotlib.pyplot as plt

plt.ion()

def _parabola_fig(b, y, members, filename, proj='Supergalactic', grid=None, stat='Mean', 
                 varname=r'$\mathbf{\tau}$', ylim=None, title=None, bin_edges=None):

    if proj == 'Galactic':
        if grid is not None:
            grid_gal = grid.transform_to('galactic')
            b = np.asarray(grid_gal.b.deg)
            m = np.isfinite(b) & np.isfinite(y)
            b = b[m]
            y = y[m]
        else:
            raise ValueError("Astropy grid must be supplied for galactic projection")
    elif proj != 'Supergalactic':
        raise ValueError("Only supergalactic and galactic projections implemented")

    popt, R2_adj, b_center, y_stat, y_se = _parabola_fit(b, y, members, stat=stat, 
                                                        bin_edges=bin_edges, 
                                                        method='wls')
    a, c = popt

    b_fit = np.linspace(-90, 90, 400)
    y_fit = _parabola(b_fit, a, c)
    label = f"{stat} {varname}"
    
    s = f"{a:0.3e}"
    mant, exp = s.split("e")
    exp = int(exp)
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(b_center, y_stat, yerr=y_se, fmt='s', capsize=5, linestyle='-', 
                 color='blue', label=label)
    
    aval = a*(180/np.pi)**2
    line1 = "Parabola Fit:"
    line2 = rf"$\mathbf{{a={aval:.2f}\ rad^{{-2}}}}$"
    plt.plot(b_fit, y_fit, color="red", label=line1 + "\n" + line2)
    plt.xlim((-90, 90))
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(f"{proj} Latitude  [deg]", fontweight='semibold', size=18)
    plt.ylabel(label, fontweight='semibold', size=18)
    plt.grid(True, which="major", ls="--")
    ax = plt.gca()  # Get the current axes
    ticks = np.arange(-90, 91, 30)  # Generates ticks from -90 to 90 in steps of 30
    tlabels = [str(tick) for tick in ticks]  # Convert each tick value to a string

    ax.set_xticks(ticks)  # Set the ticks on the x-axis
    ax.set_xticklabels(tlabels)  # Set the labels for these ticks
    for tlabel in ax.get_xticklabels() + ax.get_yticklabels():
        tlabel.set_fontweight('semibold')

    # Increase line width of the plot box
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    # Set the font size for x-axis and y-axis tick labels
    tick_label_size = 14  # Set the desired size
    for tlabel in ax.get_xticklabels():
        tlabel.set_fontsize(tick_label_size)
    for tlabel in ax.get_yticklabels():
        tlabel.set_fontsize(tick_label_size)

    ax.xaxis.grid(True, which='both', ls="-") # Major gridlines for x-axis
    ax.yaxis.grid(True, which='both', ls="-") # Both y-axis major and minor gridlines
    
    plt.title(title, y=1.04, fontweight='semibold', size=18)
    plt.legend(prop={'weight': 'semibold', 'size': 15})
    #plt.tight_layout()
    plt.savefig(filename + ".png", dpi=600)
    plt.show(block=False)
    plt.pause(0.001)
    
    return popt, R2_adj, bin_edges