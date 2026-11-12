#!/bin/bash

# change these paths for iLOFAR machine/LCU specifics
# paste the python executable from activated lofarimg environment 
PYTHON="/Users/ciaradeasymillar/anaconda3/envs/lofarimg/bin/python"  # will specify the environment here on lcu from .yml file later

# cd into the project directory so relative paths work. Will also change this for LCU
cd /Users/ciaradeasymillar/ilofar_allsky/scripts || exit 1

# Locking pattern adapted from:
# https://penguin-gym-linux.com/en/articles/tutorials/flock-file-locking
# prevents overlapping cron jobs
exec 200>/Users/ciaradeasymillar/ilofar_allsky/additional_files/pipeline.lock || exit 1
/opt/homebrew/bin/flock -n 200 || { echo "already running"; exit 1; }

# Append a heading and timestamp to the processing log 
echo >> /Users/ciaradeasymillar/ilofar_allsky/logs/pipeline_run.log # just adds an empty line between each check
echo "Beginning the check at $(date '+%Y-%m-%d %H:%M:%S')" >> /Users/ciaradeasymillar/ilofar_allsky/logs/pipeline_run.log # logs the start of a new check 

# this is the actual command to run and it defines the relative paths on the command line with \ moving onto next line
# will have to provide the paths on the lcu to the pipeline file, the directory where the observations land, and the path to the log 
"$PYTHON" \
    /Users/ciaradeasymillar/ilofar_allsky/scripts/pipeline.py \
    /Users/ciaradeasymillar/test_lgc/observations/sb195-195_x10_int_5_duration_5/ \
    /Users/ciaradeasymillar/ilofar_allsky/logs/processed_log.txt >> /Users/ciaradeasymillar/ilofar_allsky/logs/pipeline_run.log 2>&1 # this sends the output to the log 