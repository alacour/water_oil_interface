#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Partition:
#SBATCH --partition=csd_lr6_192
#
# Account:
#SBATCH --account=lr_ninjaone
#SBATCH -q condo_ninjaone


# Wall clock limit:
#SBATCH --time=40:00:00
## Run command
python submit.py
