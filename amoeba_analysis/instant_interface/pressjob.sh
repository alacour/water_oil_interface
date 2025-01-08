#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task 1
#SBATCH --time=24:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=premium

# set up for problem & define any environment variables here
python submit.py
