# this script takes xst observations and outputs a list of potentially bad antennas that can later be removed from imaging script 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time

from lofarimaging import read_acm_cube, get_station_type
from allsky_funcs import process_path, process_files

# cmd line arguments parser:
# in form: python bad_antennas.py datasource 
# output: list of bad antennas to be taken into imaging (omitted from)
# doesn't currently deal with xst files of multiple observations just the cycle but that should be alright in general to get bad ones
def parse_args():

    parser = argparse.ArgumentParser(prog='Faulty antenna flagging script',
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                      description='to identify poorly performing antennas from multiple observations')

    # positionals are the mandatory arguments!
    positionals = parser.add_argument_group("Positional arguments")

    positionals.add_argument("datasource",
                                 help="Data file/directory",
                                 type=str,
                                 ) 

    # optional arguments specified on the command line. can maybe remove this if it's a bit redundant since we aren't changing station
    optional = parser.add_argument_group("Optional arguments")
    
    optional.add_argument("-s", "--station_name",
                              help= 'Provide station name, defaults to IE613LBA',
                              dest = 'station_name',
                              type = str,
                              default = "IE613LBA"
                              )
    
    return parser.parse_args()


# main function for script
def main(): # just give the datasource as a folder of observations of xst files and it gives out the bad antennas based on that 

    start_time = time.perf_counter()

    args = parse_args() # gives the arguments from parse args to this!
    datasource = args.datasource
    print('\nDatasource:', datasource, "\n")

    if Path(datasource).is_dir():
        datasource = process_files(datasource) # list of files
    else:
        raise ValueError('Datasource must be a directory of multiple observations!!')   # still needs to be a one element list for loop!

    station_name = args.station_name
    station_type = get_station_type(station_name)
    
    cubes = [] # shape will be like (512, 192, 192)
    subbands = [] # store the subband number for plotting stuff later
    for xst_file in datasource:
        cube = read_acm_cube(xst_file, station_type)
        cubes.append(cube)

        sb = int(Path(xst_file).name.split("_sb")[1].split("_")[0])
        subbands.append(sb)

    # get the autocorr powers for all antennas within each obs
    px_all = [] # shape will be like (512, 96) ? 96 antennas for each of the 512 observations/subbands
    py_all = []
    for cube in cubes:
        cube_xx = cube[:, 0::2, 0::2]
        px = np.abs(np.diag(cube_xx[0]))   # raw autocorrelation power
        px_all.append(px)   # store the powers 

        cube_yy = cube[:, 1::2, 1::2]
        py = np.abs(np.diag(cube_yy[0]))   # raw autocorrelation power
        py_all.append(py)   # store the powers 

    # get the median power per obs
    px_all = np.array(px_all) # make it an array
    py_all = np.array(py_all)
    subbands = np.array(subbands)

    # getting rid of the noisy lower subbands to get a more accurate represnattion 
    #keep = (subbands >= 120) & (subbands <= 400)
    #subbands = subbands[keep]
    #px_all = px_all[keep]
    #py_all = py_all[keep]

    # get median
    medians_x = np.median(px_all, axis=1, keepdims=True) # get the median for each set of autocorr powers (one per obs) 
    medians_y = np.median(py_all, axis=1, keepdims=True)

    # MAD within each obs 
    mad_x = np.median(np.abs(px_all - medians_x), axis=1, keepdims=True) 
    mad_y = np.median(np.abs(py_all - medians_y), axis=1, keepdims=True) 

    #standard deviation using mad for each obs
    sigma_x = 1.4826 * mad_x
    sigma_y = 1.4826 * mad_y

    # z score
    #z = ((px_all - medians) / sigma).ravel()
    z_x = ((0.6745 * (px_all - medians_x)) / mad_x ).ravel() # modified z score from text 
    z_y = ((0.6745 * (py_all - medians_y)) / mad_y ).ravel() # modified z score from text 

    plt.hist(z_x, bins=np.linspace(-5, 5, 101), color='royalblue')
    plt.xlabel("Deviation from MAD-based sigma")
    plt.ylabel("Count")
    plt.title('X polarisation')
    plt.show()

    plt.hist(z_y, bins=np.linspace(-5, 5, 101), color='crimson')
    plt.xlabel("Deviation from MAD-based sigma")
    plt.ylabel("Count")
    plt.title('Y polarisation')
    plt.show()

    # Flag anomalous antennas within each subband
    flags_x = np.abs(px_all - medians_x) > 3.5 * sigma_x
    flag_counts_x = np.sum(flags_x, axis=0)     # how many times each antenna was flagged across all the independent observations
    flags_y = np.abs(py_all - medians_y) > 3.5 * sigma_y
    flag_counts_y = np.sum(flags_y, axis=0)  
    # this is just the total number of times across all the observations that the particular antenna fell outside of the 3sigma cut 

    n_obs_x = flags_x.shape[0] # how many observations we are looking across, might have to impose some minimum number of obs so that we can be more certain
    n_obs_y = flags_y.shape[0]

    # Fraction of subbands flagged for each antenna
    flag_fraction_x = flag_counts_x / n_obs_x  # that as a fraction of all observations
    print(flag_fraction_x[2])
    flag_fraction_y = flag_counts_y / n_obs_y

    print(f"Antenna 3, X: {100 * flag_fraction_x[3]:.4}% flagged")
    print(f"Antenna 3, Y: {100 * flag_fraction_y[3]:.4}% flagged")

    threshold = 0.5
    bad_antennas_x = np.where(flag_fraction_x > threshold)[0] # then we make the cut that if it gets flagged in more than ** of the files then it's probbaly bad
    print("\nBad antenna indices in X (flagged in more than half of observations):", bad_antennas_x, "\n")

    bad_antennas_y = np.where(flag_fraction_y > threshold)[0] # then we make the cut that if it gets flagged in more than ** of the files then it's probbaly bad
    print("\nBad antenna indices in Y (flagged in more than half of observations):", bad_antennas_y, "\n")

    print('\nX:')
    for antenna in bad_antennas_x:
        print(f'Antenna: {antenna}, Flagged in: {(flag_fraction_x[antenna]*100):.4}%') # antenna + 1 here accounts for the indexing!

    print('\nY:')
    for antenna in bad_antennas_y:
            print(f'Antenna: {antenna}, Flagged in: {(flag_fraction_y[antenna]*100):.4}%') # antenna + 1 here accounts for the indexing!


    antenna_results_path = Path('/Users/ciaradeasymillar/test_lcu/results/antennas')
    antenna_results_path.mkdir(parents=True, exist_ok=True)

    # write the bad antennas to a txt file
    bad_antennas_file = open(str(antenna_results_path / 'bad_antennas.txt'), 'w')
    bad_antennas_file.write(" ".join(str(ant) for ant in bad_antennas_x)) # makes list a string
    bad_antennas_file.close()

    # X 
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(96), flag_fraction_x, color='royalblue')
    ax.axhline(threshold, color='crimson', linestyle='--')
    ax.set_xlabel("Antenna number")
    ax.set_ylabel("Fraction of subbands flagged")
    ax.set_title("Persistent antenna power outliers in X")
    fig.savefig(str(antenna_results_path / 'outliers_X.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Y
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(96), flag_fraction_y, color='royalblue')
    ax.axhline(threshold, color='crimson', linestyle='--')
    ax.set_xlabel("Antenna number")
    ax.set_ylabel("Fraction of subbands flagged")
    ax.set_title("Persistent antenna power outliers in Y")
    fig.savefig(str(antenna_results_path / 'outliers_Y.png'), dpi=300, bbox_inches='tight')
    plt.show()

    subband_power_x = np.median(px_all, axis=1)
    subband_power_y = np.median(py_all, axis=1)

    # X
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subbands, subband_power_x, marker='.', linestyle='-', color='royalblue')
    ax.set_xlabel("Subband number")
    ax.set_xlabel("Subband index")
    ax.set_ylabel("Median autocorrelation power")
    ax.set_title("Median antenna power across subbands in X")
    ax.set_ylim(-1,0.4e9)
    fig.savefig(str(antenna_results_path / 'med_autocorr_power_X.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # x
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subbands, subband_power_y, marker='.', linestyle='-', color='royalblue')
    ax.set_xlabel("Subband number")
    ax.set_xlabel("Subband index")
    ax.set_ylabel("Median autocorrelation power")
    ax.set_title("Median antenna power across subbands in Y")
    ax.set_ylim(-1,0.4e9)
    fig.savefig(str(antenna_results_path / 'med_autocorr_power_Y.png'), dpi=300, bbox_inches='tight')
    plt.show()

    script_runtime = time.perf_counter() - start_time
    print(f"runtime of script: {script_runtime:.2f}")

if __name__ == "__main__":
    main() # so that it executes main 