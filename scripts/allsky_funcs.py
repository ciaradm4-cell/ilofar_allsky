from filesplit.split import Split 
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import sys
import os
import matplotlib.patheffects as path_effects
import re
import imageio
"""Functions for working with LOFAR single station data"""
import matplotlib.gridspec as gridspec
import os
import datetime
from typing import List, Dict, Tuple, Union
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from packaging import version
import tqdm
import h5py

import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.coordinates import SkyCoord, GCRS, EarthLocation, AltAz, get_sun, get_body
import astropy.units as u
from astropy.time import Time

def get_moon(time, location=None, ephemeris=None):
    return get_body('moon', time, location, ephemeris)

import lofargeotiff
from lofarantpos.db import LofarAntennaDatabase
import lofarantpos

#from .maputil import get_map, make_leaflet_map
#from .lofarimaging import nearfield_imager, sky_imager, skycoord_to_lmn, subtract_sources
#from .hdf5util import write_hdf5


# splits an xst file that contains more than one image up into a directory containing the individual image files for processing 
def split_file(datasource):
    #print(datasource)
    file_size = os.path.getsize(datasource)
    #print(file_size)

    if file_size == 301989888: # temporary change check here
        print(f'file is ACC file with size {file_size}.... splitting accordingly')

        filename = os.path.basename(datasource)
        print(filename)
        subfolder = "_".join(filename.split("_")[:2])  # date_time
        #print(subfolder)

        
        output_dir = os.path.join(os.path.dirname(datasource), 'ACC_files', subfolder) + '/'
        print('output directory:', output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            split = Split(inputfile=datasource, outputdir=output_dir)
            split.bysize(size=589824)
        
        datasource = glob.glob(output_dir + '*.dat') # now the datasource is the new directory of split files so the format holds from og code
        datasource.sort()

    else:
        filename = os.path.basename(datasource)
        subfolder = "_".join(filename.split("_")[:2])  # date_time

        output_dir = os.path.join(os.path.dirname(datasource), 'split_data', subfolder) + '/'
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir) # make the directory if it doesnt exits 

            split = Split(inputfile= datasource, outputdir= output_dir) # splits the file up by size seems to work well siince size is consistent 
            split.bysize(size=589824)

        datasource = glob.glob(output_dir + '*xst[_.]*dat') # now the datasource is the new directory of split files so the format holds from og code
        datasource.sort()
        print('datasource:',datasource)

    return datasource


def process_path(datasource): #converts '\' to '/' in paths pasted in from windows PCs

    # checks size of file
    file_size = os.path.getsize(datasource)
    print(file_size)

    # if its bigger than this then it containes more than one observation so we need to split it up
    if file_size > 589824:
       print(f'\ndatasource is one xst file with multiple images, size {file_size} bytes')
       print('Caution: for accurate image timestamps, provide integration time using -int!')
       datasource = split_file(datasource)
    else:
       datasource = [datasource]

    return datasource


def process_files(datasource):
    filenames = glob.glob(datasource + '*xst[_.]*dat') # now this works for both types of directories (those from an actual directory and those from directories made from splitting an xst file)
    #print(filenames)
    filenames.sort()
    num_files = len(filenames)
    print(str(num_files) + ' xst data files to be processed....')

    if num_files == 0:
       print('Glob failed')
       print('If you are working with a directory, you must add a trailing / at the end of the path on the command line')
    else:   
        return filenames


def plot_correlations(cube_xx, cube_yy, visibilities):
   # Check how the correlations look
    fig1, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8,3))

    ax0.imshow(cube_xx[0,:,:].real, aspect='auto', cmap=cm.jet)
    ax0.set_xlabel('Antenna number')
    ax0.set_ylabel('Antenna number')
    ax0.set_title('XX')
    ax1.imshow(cube_yy[0,:,:].real, aspect='auto', cmap=cm.jet)
    ax1.set_title('YY')
    ax1.set_xlabel('Antenna number')
    ax1.set_ylabel('Antenna number')
    ax2.imshow(visibilities.real, aspect='auto', cmap=cm.jet)
    ax2.set_title('I')
    ax2.set_xlabel('Antenna number')
    ax2.set_ylabel('Antenna number')
    fig1.tight_layout()


def plot_antennas(station_xyz):

   fig, ax = plt.subplots(figsize=(3, 3))
   ax.scatter(station_xyz[:, 0], station_xyz[:, 1], s=2, facecolor='k');
   ax.set_title("Antenna positions");
   ax.set_aspect(1);


def plot_baselines(baselines):
   plt.figure(figsize=(3,3))
   plt.scatter(baselines[:,:,0], baselines[:,:,1], facecolor='k', edgecolor='k', s=1, alpha=0.1);


def make_gif(frames, out_dir='results/gifs', name=None, duration=500):
# Source - https://stackoverflow.com/a/35943809
# Posted by Almar, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-07, License - CC BY-SA 4.0
# adapted function 

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"{Path(frames[0]).stem}_to_{Path(frames[-1]).stem}"

    path_to_gif = out_dir / f"{name}.gif"
    ims = [imageio.imread(f) for f in frames]
    imageio.mimsave(path_to_gif, ims, duration=duration, loop=0)

    print('GIF saved as', path_to_gif)

    return path_to_gif


# got ai to make plot correct here as was having issues 
def make_sky_plot(
    image: np.ndarray,
    marked_bodies_lmn: Dict[str, Tuple[float, float, float]],
    title: str = "Sky plot",
    subtitle: str = "",
    fig: Figure = None,
    label: str = None,
    savepath: str = None,
    galactic_plane_lmn=None,
    **kwargs) -> Figure:
    """
    Make a sky plot out of an array with data.

    Args:
        image: numpy array (two dimensions with data)
        marked_bodies_lmn: dict with objects to annotate
                           (values should be lmn coordinates)
        title: Title for the plot
        subtitle: Subtitle for the plot
        fig: existing figure object to be reused
        label: unique label for axes
               (only relevant for making animations)
        savepath: path to save the figure, e.g. "sky_plot.png"
        **kwargs: other options to be passed to plt.imshow
                  (e.g. vmin, vmax)

    Returns:
        Updated figure
    """

    if fig is None:
        fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1, label=label)

    plt.get_current_fig_manager().resize(1024, 900)

    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Colours
    fore_color = 'white'
    axis_color = '0.4'

    plt.rcParams["text.color"] = fore_color

    # Main circular sky boundary
    circle1 = Circle(
        (0, 0),
        1.0,
        edgecolor='white',
        linewidth=0.7,
        fill=False,
        facecolor='none'
    )

    ax.add_artist(circle1)

    # Plot image
    cimg = ax.imshow(
        image,
        origin='lower',
        cmap=cm.jet,
        extent=(1, -1, -1, 1),
        clip_path=circle1,
        clip_on=True,
        **kwargs
    )

    # Keep l-m plane circular
    ax.set_aspect('equal')

    # Colourbar outside plot
    cax = inset_axes(
        ax,
        width="4%",
        height="95%",
        loc="center left",
        bbox_to_anchor=(1.05, 0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    colourbar = fig.colorbar(
        cimg,
        cax=cax,
        orientation="vertical",
        format="%.2e"
    )

    # Colourbar styling
    colourbar.ax.tick_params(colors=axis_color, labelsize=9)
    colourbar.outline.set_edgecolor(axis_color)

    # Axis limits
    ax.set_xlim(1, -1)
    ax.set_ylim(-1, 1)

    # Major ticks only
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.xaxis.set_major_formatter(
        FormatStrFormatter('%.1f')
    )

    ax.set_yticks(np.arange(-1, 1.1, 0.5))
    ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.1f')
    )

    # Make major ticks and tick labels grey
    ax.tick_params(
        axis='both',
        which='major',
        colors=axis_color,
        length=5, 
        labelsize=9
    )

    # Make square axes frame grey
    for spine in ax.spines.values():
        spine.set_color(axis_color)

    # Axis labels
    ax.set_xlabel(
        r'$\ell$',
        fontsize=12,
        color=axis_color
    )

    ax.set_ylabel(
        r'$m$',
        fontsize=12,
        color=axis_color
    )

    # Title
    ax.text(
        0.5,
        1.08,
        title,
        fontsize=13,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        color='white'
    )

    # Subtitle
    ax.text(
        0.5,
        1.02,
        subtitle,
        fontsize=9,
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        color='white'
    )

    # Mark astronomical bodies
    for body_name, lmn in marked_bodies_lmn.items():

        ax.plot(
            [lmn[0]],
            [lmn[1]],
            marker='*',
            color='black',
            mew=0.5,
            mec='white',
            ms = 7
        )

        ax.annotate(
            body_name,
            (lmn[0], lmn[1]),
            color='black',
            fontsize='9',
            xytext=(8, 5),
            textcoords="offset points",
            path_effects=[
                path_effects.withStroke(
                    linewidth=1,
                    foreground='white'
                )
            ]
        )

    # Compass directions
    ax.text(
        0.9, 0, 'E',
        horizontalalignment='center',
        verticalalignment='center',
        color='white',
        fontsize=13
    )

    ax.text(
        -0.9, 0, 'W',
        horizontalalignment='center',
        verticalalignment='center',
        color='white',
        fontsize=13
    )

    ax.text(
        0, 0.9, 'N',
        horizontalalignment='center',
        verticalalignment='center',
        color='white',
        fontsize=13
    )

    ax.text(
        0, -0.9, 'S',
        horizontalalignment='center',
        verticalalignment='center',
        color='white',
        fontsize=13
    )

    # Save figure if requested
    if savepath is not None:
        fig.savefig(
            savepath,
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )

    return fig

