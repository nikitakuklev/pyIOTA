__all__ = ['tbt_load']

import gc
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

#__import__('tables')  # <-- import PyTables; __import__ so that linters don't complain, used for BLOSC codecs
import h5py


def tbt_load(data_path: Optional[Path] = None,
             info_path: Optional[Path] = None,
             dpath: Optional[Path] = None,
             root_name: str = 'test',
             watchpoint: str = 'CHRG',
             observation_number: int = 1,
             silent: bool = False) -> pd.DataFrame:
    """
    Loads turn-by-turn data from elegant watchpoint files
    :param data_path: Full file path, overrides auto naming
    :param info_path: Full file path, overrides auto naming
    :param dpath: Directory with files, to use with default file format
    :param root_name:
    :param watchpoint:
    :param observation_number:
    :param silent: No debug printing
    :return:
    """
    if not (data_path and info_path):
        storeinfo = '{}-trackall-{}-info-obs{:04d}.hdf5'.format(root_name, watchpoint, observation_number)
        storedata = '{}-trackall-{}-data-obs{:04d}.hdf5'.format(root_name, watchpoint, observation_number)
        info_path = dpath/storeinfo
        data_path = dpath/storedata

    df_info = pd.read_hdf(str(info_path))
    with h5py.File(str(data_path), 'r') as f:
        if not silent: print('Reading TBT (HDF5):', f)
        cols = f.attrs['columns'].split(',')
        df_data = pd.DataFrame(columns=cols, dtype=np.dtype(object))
        for p in f:
            if f[p].shape is None:
                df_data.loc[int(p), :] = [np.array([])] * 7
            else:
                df_data.loc[int(p), :] = list(f[p][()].T)

    df = pd.concat([df_info, df_data], axis=1)
    df.sort_index(inplace=True)
    del df_info
    del df_data
    gc.collect()
    return df
