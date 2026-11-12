import glob
import sys
import argparse
import numpy as np
from pathlib import Path
from filesplit.split import Split 
import subprocess

# this script takes an obs directory and processing log as inputs
# reads the log to see what files have already been processed and identifies from the obs directory which files are new
# outputs how many new files exist and if there's X files, it batches them

def parse_args():

    # says what the script does
    parser = argparse.ArgumentParser(prog="Processed files log checker",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Looks at the log of processed files and uses this to check the output observations directory for new files."
                                     )

    # mandatory args
    positionals = parser.add_argument_group("Positional arguments")

    # where the raw XST files land
    positionals.add_argument("obs_directory",
                             help="Directory of observations as they arrive",
                             type=str,
                             ) 

    # log file of previously processed files
    positionals.add_argument("processed_files_log",
                                 help="Processed files .txt file log",
                                 type=str,
                                 )

    return parser.parse_args()


# main function
def main():

    args = parse_args()

    scripts_dir = Path(__file__).resolve().parent
    project_dir = scripts_dir.parent

    imaging_script = scripts_dir / "lofar_allsky_imaging.py"
    caltable_dir = project_dir / "additional_files" / "caltables"
    bad_antenna_path = project_dir / "results" / "antennas" / "bad_antennas.txt"

    # get args from command line
    obs_dir = args.obs_directory
    log = args.processed_files_log

    # glob all the observations together from output directory and sort them in order
    obs_dir_glob = glob.glob(str(Path(obs_dir) / "**" / "*_xst.dat"), recursive=True)   # this allows for the subdirectories within the main observations directory 
    obs_dir_glob.sort()

    # read the processed files log and store them
    with open(log, 'r') as files:
        processed_files = [line.strip() for line in files]   

    # converts them to a set, more efficient
    processed_files = set(processed_files) 
    
    # empty list to fill with the observations that haven't been processed yet 
    not_processed_obs = []

    # for each observation in the directory, check if it has already been processed:
    for obs_path in obs_dir_glob:

        # if the file isn't in the log of already processed files then list it so it can be processed further down 
        if Path(obs_path).name not in processed_files:
            not_processed_obs.append(obs_path)

    # print results where flush=True was implemented so that this message gets written immediately before processing instead of buffering causing it to be printed last after processing
    print(f"\nTotal XST files/observations: {len(obs_dir_glob)}", flush=True)
    print(f"Previously Processed: {len(processed_files)}", flush=True)
    print(f"New files: {len(not_processed_obs)}", flush=True)

    # filter out faulty antenna if there is any to remove from processing
    bad_antenna_file = open(bad_antenna_path)
    bad_antennas_str = bad_antenna_file.read()
    print('bad antennas:', bad_antennas_str)

    # decide how many files to be processed at once
    batch_size = 5

    # run the imaging script for the not_processed files. The cron job needs ignores if this is currently running (it uses flock to wait until a whole batch is processed)
    if len(not_processed_obs) >= batch_size:    # imposing a minimum batch size for processing for efficiency 

        # select a batch of the unprocessed files
        batch = not_processed_obs[:batch_size] # select a batch of the unprocessed files

        # Use the subprocess module as a tool to run other scripts/commands from within code 
        if bad_antennas_str:
            print('omitting antennas from imaging:', bad_antennas_str)
            bad_antennas = bad_antennas_str.split() # split into individual numbers 
            subprocess.run([sys.executable, imaging_script, *batch, "-c", caltable_dir , "-no_im", "-b", *bad_antennas], check=True)
        else:  
            print('No antennas to remove')
            subprocess.run([sys.executable, imaging_script, *batch, "-c", caltable_dir, "-no_im"], check=True)   # *batch uses * the unpacking operator which takes the file paths and passes each as a seperate cmd line arg

if __name__ == "__main__":
    main()