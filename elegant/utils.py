__all__ = ['load_data_tbt']

import gc
__import__('tables')  # <-- import PyTables; __import__ so that linters don't complain
import h5py
import numpy as np
import pandas as pd

# This is a legacy function, deprecated

def load_data_tbt(fpath, root_name='test', watchpoint='CHRG', observation_number=1, silent=False):
    storeinfo = '{}-trackall-{}-info-obs{:04d}.hdf5'.format(root_name, watchpoint, observation_number)
    storedata = '{}-trackall-{}-data-obs{:04d}.hdf5'.format(root_name, watchpoint, observation_number)
    df_info = pd.read_hdf(str(fpath / storeinfo))
    with h5py.File(str(fpath / storedata), 'r') as f:
        if not silent: print('Reading TBT (HDF5):', f)
        cols = f.attrs['columns'].split(',')
        df_data = pd.DataFrame(columns=cols, dtype=object)
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
