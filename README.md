# pyIOTA

Accelerator control, modelling, and data analysis framework

## Features

pyIOTA is a mix everything I found useful in experimental beam dynamics studies - glue I/O and logic to talk with various accelerator codes, algorithms for optics and TBT data analysis, adapters for controlling accelerators and storing experimental data, and other small things.

It supports:
* Lattice input and output
    - Elegant/MADX/6dsim to native OCELOT-like structure and back
* Linear optics and thin tracking
    - Wraps and extends OCELOT core with utility methods, supports more magnet types
* TBT and orbit data analysis
    - Preprocessing (anomaly detection, thresholding, etc.)
    - Phase space, tunes, and optics functions recovery
    - Mode decomposition (SVD, ICA, etc.)
    - Nonlinear analysis
* Simulation
    - Taskfile generation - DA, FMA, mom. aperture, and other types of studies
    - Parameter space scans and cluster job management through `dask`
    - Integration with MOGA and other custom optimizers
* Control
    - System-agnostic, object oriented interface that uses native lattice format, does name/value translation, and knob generation/storage/math
    - Converters to real device commands - currently supports Fermilab network via ACNET
    - Easy to use with Jupyter notebooks and standard data science setups for live data analysis
* Utilities
    - Data storage, custom math types (i.e. bounded, wrap-around float), plotting functions, etc.

## Getting Started

### Prerequisites

- Python 3.7+
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `httpx` (ACNET only)
- `ACNETProxy` (https://github.com/nikitakuklev/ACNET-Proxy) or Fermilab controls library jar file  (ACNET only)
- `SDDS Python 3.7` (SDDS input/output only, available at https://www.aps.anl.gov/Accelerator-Operations-Physics/Software)
- `pymadx3` (https://github.com/nikitakuklev/pymadx3) or another TFS parser (MADX only)

The suggested development environment is PyCharm or VSCode along with the Anaconda python distribution.

### Installing

##### Linux/Windows
Clone the repo and add to your PYTHONPATH. However, it is often easier to just use `sys.path.insert` from within your code prior to import.

```
$ git clone https://github.com/nikitakuklev/pyIOTA.git
```

##### OS X

Untested, but should work the same

### Usage

There are a few example notebooks in `examples` folder showing various usages. Also, most methods are fairly well documented. If you are interested in specific code associated with studies and publications, please contact the author.

## Contributing

Should anyone be interested in contributing, pull requests are welcome. The coding style follows standard PEP8, with reST docstrings. For debugging on internal controls networks, PyCharm remote deployment or VSCode Remote SSH features are extremely convenient. Please make sure tests pass before submitting.

## License
All code is GPLv3, except as indicated in the files - see the [LICENSE](LICENSE) file for details.


## Disclaimer
Authors take absolutely no responsibility for all the horrible things that can happen due to usage of this code. In particular, not for the sign of certain knob math being flipped and messing up a whole shift of data collection...not that such a thing ever happened... 

