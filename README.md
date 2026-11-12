# iLOFAR All-Sky Imaging

An automated all-sky imaging pipeline developed for my final-year physics project at Trinity College Dublin, using observations from the Irish LOFAR station.

The pipeline checks an observation directory for new XST files and processes them in batches. It applies calibration, excludes flagged antennas and autocorrelations, and produces all-sky images. A processed-file log prevents repeat processing, while a Bash wrapper allows scheduled runs through cron.

## Contents

- `scripts/`: imaging, processing and antenna-flagging scripts, plus supporting code.
- `additional_files/`: calibration tables and the Conda environment file.
- `results/antennas/`: the antenna exclusion list.

Raw observations, generated images and runtime logs are excluded from this repository.

## Status

Currently being prepared for testing on the iLOFAR LGC. (Will add stacking and temporal difference imaging after testing

## Acknowledgements

This project builds on the `lofarimaging` package and uses `lofarantpos`, Astropy, NumPy and other scientific Python tools. The included `lofarimaging` source is third-party code; attribution and details of local modifications will be documented before public release.
