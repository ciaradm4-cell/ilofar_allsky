#!/bin/bash

PYTHON="/Users/ciaradeasymillar/anaconda3/envs/lofarimg/bin/python"

cd /Users/ciaradeasymillar/test_lcu/scripts || exit 1

# trigger the observation to take one scan across the subbands - this calls the dedicated checker scan acquisition script
bash /Users/ciaradeasymillar/test_lcu/scripts/jmcc_xst_antenna_checker_scan.sh 3 50 419 5 5 1 ciara

# obtain the scan directory from the run!
# cat reads the txt file and prints it's contents and the $() captures that instead of printing it and then the = stores it as a var
SCAN_DIR=$(cat scan_directory.txt)

# passing that directory to the antenna checker so it runs on the directory that the scan created
"$PYTHON" bad_antennas.py "$SCAN_DIR"
