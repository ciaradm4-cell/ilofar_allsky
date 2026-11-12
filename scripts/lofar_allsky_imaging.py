import time
import datetime
import argparse
import numpy as np
import lofarantpos
from pathlib import Path
import astropy.units as u 
from packaging import version
from astropy.time import Time
import matplotlib.pyplot as plt
from filesplit.split import Split 
from matplotlib.patches import Rectangle
from lofarantpos.db import LofarAntennaDatabase
from astropy.coordinates import SkyCoord, GCRS, EarthLocation, AltAz, get_sun

# functions imports from other files 
from lofarimaging import (freq_from_sb, read_acm_cube, sky_imager, get_station_type,
    skycoord_to_lmn, apply_calibration, get_full_station_name, get_station_xyz)
from lofarimaging.lofarimaging import subtract_sources
from allsky_funcs import (process_files, process_path, plot_correlations, plot_antennas,
    plot_baselines, make_sky_plot, make_gif)

assert version.parse(lofarantpos.__version__) >= version.parse("0.4.0")
print("\nimports ok")

# This script takes in raw LOFAR LBA XST observations and produces all-sky images
# The input XST can be one file with multiple observations or single observation XST files 

# arguments parser from command line to variables
def parse_args():

    # what the programme does:
    parser = argparse.ArgumentParser(prog="LBA All-Sky Imager for iLOFAR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="runs the all-sky imaging pipeline for iLOFAR LBA observations."
                                     )
 
    # positionals are the mandatory arguments:
    positionals = parser.add_argument_group("Positional arguments")

    # provide the datasource (XST obs)
    positionals.add_argument("datasource",
                             help="Data file/directory",
                             type=str,
                             nargs='+',
                             ) 

    # optional arguments specified on the command line
    optional = parser.add_argument_group("Optional arguments")
    
    # This is required if working with multiple observation XST files so that the timestamp on each image can be corrected
    optional.add_argument("-int", "--integration_time",
                          help='option -int gives the exposure/integration time (needed for xst files with multiple observations embedded)',
                          dest="int_time",
                          type=int,
                          default=False,
                          required=False,
                          )
    
    # if there is calibration data it can pe provided here as a directory 
    optional.add_argument("-c", "--calibration_table_dir",
                          help='Calibration data directoryto be provided as a path',
                          dest="caltable_dir", 
                          type=str,
                          default=False,
                          required=False,
                          )

    # if we have multiple images and want to make a gif out of them in time series then call this
    optional.add_argument("-g", "--gif", 
                          help='option -g makes a gif out of consecutive constructed images',
                          action='store_true',
                          default=False,
                          required=False
                          )

    # plots the correlation matrix 
    optional.add_argument("-pc", "--plot_correlations",
                          help='Plots the correlation matrix for a given observation',
                          action='store_true',
                          default=False,
                          required=False
                          )

    # plots the baselines for the observation
    optional.add_argument("-pb", "--plot_baselines",
                          help='Plots the baselines (u,v) for a given observation',
                          action='store_true',
                          default=False,
                          required=False
                          )

    # plots the antenna positions for the observation 
    optional.add_argument("-pa", "--plot_antennas",
                          action='store_true',
                          help='Plost the antenna positions (x,y) for a given observation',
                          default=False,
                          required=False
                          )

    # change RCU mode (standard for these LBA obs is 3)
    optional.add_argument("-r", "--rcu_mode",
                          help= 'Provide rcu mode, defaults to 3',
                          dest = 'rcu_mode',
                          type = str,
                          default = "3"
                          )

    # Stops pop-up of each image and only saves instead - in nightly automation this should always be provided 
    optional.add_argument("-no_im", "--no_image",
                          action='store_true',
                          help='Do not plot each image',
                          default=False,
                          required=False
                          )

    # if there is a list of antennas to be omitted, provide it here as a txt file 
    optional.add_argument("-b", "--bad_antennas",
                          dest='bad_antennas',
                          help='Takes a provided list of bad antennas list and omits them from imaging',
                          nargs='+', # allows it to take more than one arg!
                          type=int,
                          default=[], # empty list by default so it doesnt crash 
                          required=False
                          )
 
    return parser.parse_args()


# main function of the script
def main():

    start_of_script = time.perf_counter()
    args = parse_args() # gives the arguments from parse args to main

    datasource = args.datasource 
    caltable_dir = args.caltable_dir
    rcu_mode = args.rcu_mode
    station_name = "IE613LBA"
    bad_antennas = np.array(args.bad_antennas)
    print('\nDatasource:', datasource)

    # station specific data 
    rcu_mode = rcu_mode.replace("sparseeven", "sparse_even").replace("sparseodd", "sparse_odd")
    station_type = get_station_type(station_name)
    station_name = get_full_station_name(station_name, rcu_mode)

    # grabs positional information 
    db = LofarAntennaDatabase()
    station_xyz = get_station_xyz(station_name, rcu_mode, db)[0]

    # if bad antennas list is provided, mask these out before imaging 
    if bad_antennas.size > 0:
        mask = ~np.isin(np.arange(len(station_xyz)), bad_antennas)
        station_xyz = station_xyz[mask]
        print(f'\nThe following antennas were manually omitted from imaging: {bad_antennas}')

    # grab the baselines from the xyz coords of each antenna 
    baselines = station_xyz[:, np.newaxis, :] - station_xyz[np.newaxis, :, :]

    # harcoded variables
    extent = [-150, 150, -150, 150]
    npix_l, npix_m = 131, 131
    pixels_per_metre = 0.5
    timestep = 0
    height = 1.5

    print(f"""      
        Station: {station_name}
        Station type: {station_type}
        Timestep: {timestep}
        Sky image size: {npix_l:d} x {npix_m:d}
        Ground image extent: {extent[1]-extent[0]:d} x {extent[3]-extent[2]:d} m
        Ground height: {height} m\
        """)

    # make a results directory for this if it doesn't already exist
    results_dir = Path(__file__).resolve().parent.parent / "results" / "allsky_images"
    results_dir.mkdir(parents=True, exist_ok=True)

    # optional additional plots
    if args.plot_antennas:
        plot_antennas(station_xyz)
            
    if args.plot_baselines:
        # For every antenna, calculate the distance between it and every other antenna
        plot_baselines(baselines)

    # open tthe log to write the names of the processed files to. a is for append here instead of overwriting the file
    log_path = Path(__file__).resolve().parent.parent / "logs" / "processed_log.txt"
    log = open(log_path, "a")

    # initialising empty lists to store values 
    frames = []
    sub_frames = []
    images = []

     # start timing loop here
    start_of_loop = time.perf_counter()

    # for file in datasource:
    for original_file in datasource:

        # if there is a provided integration time, parse it and if not, try parse it from the filepath 
        if args.int_time:
            int_time = args.int_time
        else:
            int_time = int(Path(original_file).parent.name.split("_int_")[1].split("_")[0])   

        # process path returns a list of all of the inidvidual files in the datasource. This is what deals with splitting multi-obs XST files
        for file in process_path(original_file):

            # parsing the basic data from the filename
            filename = file.split('/')[-1]
            print(f"\nCurrent file: {filename}")

            file_comps = filename.split('_')      # get the individual components stored in the filename
            subband =  int(file_comps[2].replace('sb', ''))     # make integer for later handling
            obsdatestr = file_comps[0]      # YYYYMMDD of the observation (always at start of filename)

            # if multi-obs XST file, use provided int time to correct each timestamp, otherwise parse timestamp directly from filename
            if '_xst_' in file:
                im_num = int(file.split('_xst_')[-1].replace('.dat', '')) - 1
                obstimestr =  (datetime.datetime.strptime(file_comps[1], '%H%M%S') + datetime.timedelta(seconds=im_num * int_time)).strftime('%H%M%S')# HHMMSS of the observation - always second part of filename
                obstime = datetime.datetime.strptime(obsdatestr + ":" + obstimestr, '%Y%m%d:%H%M%S')   # different structure for later use
            else:
                obstimestr = file_comps[1]      # HHMMSS of the observation - always second part of filename
                obstime = datetime.datetime.strptime(obsdatestr + ":" + obstimestr, '%Y%m%d:%H%M%S')    # different structure for later use

            # get freq from subband 
            freq = freq_from_sb(subband, rcu_mode=rcu_mode) # Which slice in time to visualise
        
            # Confirm the data has been read correctly
            print(f"""    
            Subband: {subband}
            Timestamp: {obstime}
            Frequency: {freq / 1e6:.2f} MHz
            """)

            # read the acm binary data cube and output a 3D cube of complex numbers with indices [time slots, rcu, rcu]
            cube = read_acm_cube(file, station_type)

            # if calibration data given then apply this to the cube
            if caltable_dir:
                cube, _ = apply_calibration(cube, station_name, rcu_mode, subband, caltable_dir=caltable_dir)
            
            # Split the matrix into the XX and YY polarisations 
            cube_xx = cube[:, 0::2, 0::2]
            cube_yy = cube[:, 1::2, 1::2]

            visibilities_all = cube_xx + cube_yy
            visibilities = visibilities_all[timestep]   # Stokes I for specified timestep

            # mask corresponding visibilities if omitting antennas 
            if bad_antennas.size > 0:
                visibilities = visibilities[mask][:, mask] # mask rows and columns

            # Remove autocorrelations from observations by zero-ing the diagonal 
            np.fill_diagonal(visibilities, 0)

            # optional additional plots code
            if args.plot_correlations:
                plot_correlations(cube_xx, cube_yy, visibilities)

            # Determine positions of Cas A and Cyg A for plotting and source subtraction 
            obstime_astropy = Time(obstime)
            station_earthlocation = EarthLocation.from_geocentric(*(db.phase_centres[station_name] * u.m))
            zenith = AltAz(az=0 * u.deg, alt=90 * u.deg, obstime=obstime_astropy,
                        location=station_earthlocation).transform_to(GCRS(obstime=obstime_astropy))

            # default objects to mark, can be altered 
            marked_bodies = {
                'Cas A': SkyCoord(ra=350.85*u.deg, dec=58.815*u.deg),
                'Cyg A': SkyCoord(ra=299.868*u.deg, dec=40.734*u.deg),
                'Sun': get_sun(obstime_astropy)
            }

            # coordinate transformation
            marked_bodies_lmn = {}
            for body_name, body_coord in marked_bodies.items():
                if body_coord.transform_to(AltAz(location=station_earthlocation, obstime=obstime_astropy)).alt > 0:
                    marked_bodies_lmn[body_name] = skycoord_to_lmn(marked_bodies[body_name], zenith)

            # make all sky image
            stem = Path(file).stem   # eg 20260924_233824_sb279_xst_0001
            path_to_frame = str(results_dir / f'{stem}_skyplot.png')    # where the image will be saved to 

            # pass visibilities to imager and append the image to list for stacking later if needed 
            start_imaging = time.perf_counter()
            img = sky_imager(visibilities, baselines, freq, npix_l, npix_m) 
            images.append(img)

            # timing 
            imaging_runtime = time.perf_counter() - start_imaging
            print(f"Sky imaging took {imaging_runtime:.3f} seconds")

            # generate figure to be plotted and saved 
            start_fig = time.perf_counter()
            fig = make_sky_plot(img, marked_bodies_lmn, title=f"Sky image for {station_name}",
                        subtitle=f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:19]}", savepath=path_to_frame)
   
            # if we are making a gif, append the paths to each frame to a list for the gif-making function to grab later 
            if args.gif:
                frames.append(path_to_frame) 

            # if command to not show image, don't generate a pop-up
            if not args.no_image:
                plt.show()  

            plt.close(fig)

            # timing 
            fig_runtime = time.perf_counter() - start_fig
            print(f"Figure generation imaging took {fig_runtime:.3f} seconds")

            # savepath of subtracted image
            path_to_sub_frame = str(results_dir / f'{stem}_subskyplot.png')

            # make the subtracted image
            start_subtraction = time.perf_counter()
            visibilities_subtracted = subtract_sources(visibilities, baselines, freq, marked_bodies_lmn)
            img_subtracted = sky_imager(visibilities_subtracted, baselines, freq, npix_l, npix_m)
            
            # timing 
            subtraction_runtime = time.perf_counter() - start_subtraction
            print(f"Sky subtraction took {subtraction_runtime:.3f} seconds")

            # plotting and fixed scale 
            lo, hi = np.nanquantile(img_subtracted, [0.05, 0.95])
            fig = make_sky_plot(img_subtracted, marked_bodies_lmn, title=f"Sky image for {station_name}",
                            subtitle=f"SB {subband} ({freq / 1e6:.1f} MHz), {str(obstime)[:19]}",
                            vmin=lo, vmax=hi, savepath=path_to_sub_frame)

            # save path for gif making 
            if args.gif:
                sub_frames.append(path_to_sub_frame) 

            if not args.no_image:
                plt.show()  

            plt.close(fig) 

        # after each file, save the filename to the log so that files are not processed more than once
        log.write(Path(original_file).name + "\n") 
        log.flush()

    log.close()
    print("File written to log file")

    if args.gif:
            make_gif(frames=frames)
            make_gif(frames=sub_frames)

    end_of_loop = time.perf_counter()
    loop_runtime = end_of_loop - start_of_loop

    # timing results 
    print(f"\nTotal runtime for {len(datasource)} files: {loop_runtime:.2f} seconds")
    print(f"Average runtime per file: {loop_runtime / len(datasource):.2f} seconds")
    script_runtime = time.perf_counter() - start_of_script
    print(f"runtime of entire script: {script_runtime:.2f}")

if __name__ == "__main__":
    main()