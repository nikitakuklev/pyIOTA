"""
pyIOTA accelerator control and analysis framework
"""

__version__ = '0.9.5'
__author__ = "Nikita Kuklev"

# General imports - without these, most things will not work
import sys
import os
import logging

logging.basicConfig(  # format='[%(asctime)s] {%(name)s:%(funcName)s:%(lineno)d} %(levelname)s - %(message)s',
    #format='[%(asctime)s][%(funcName)s:%(lineno)d}] %(levelname)s - %(message)s',
    format='[%(levelname)-5.5s][%(asctime)s][%(name)10.10s:%(lineno)4s] %(message)s',
    level=logging.INFO,
    stream=sys.stdout)

# Detect number of cores and limit to 8
N_CORES_MAX = 8
n_cores = os.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores if n_cores <= N_CORES_MAX else N_CORES_MAX)

# import pyIOTA.lattice as lat

# %(module)s
# %(pathname)s
