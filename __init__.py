"""
pyIOTA accelerator control and analysis framework
"""

__version__ = '0.7.5'
__author__  = "Nikita Kuklev"

# General imports - without these, most things will not work
import sys

import numpy as np
import pandas as pd
import logging

logging.basicConfig(format='[%(asctime)s] {%(name)s:%(funcName)s:%(lineno)d} %(levelname)s - %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

#%(module)s
#%(pathname)s