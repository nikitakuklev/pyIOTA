__all__ = ['compare_twiss_to_box', 'merge_twiss_with_box']

import gc
__import__('tables')  # <-- import PyTables; __import__ so that linters don't complain
import h5py
import numpy as np
import pandas as pd

def merge_twiss_with_box(box, twi):
    df_t = box.twiss_df()
    df_t['name'] = df_t['name'].shift(1).str.lower()
    twi['ElementName'] = twi['ElementName'].str.lower()

    df_tc = df_t.drop_duplicates('name')
    twic = twi.drop_duplicates('ElementName')

    df_m = df_tc.merge(twic, right_on=['ElementName'], left_on=['name'])
    assert all(np.isclose(df_m.s_x, df_m.s_y))
    return df_m

def compare_twiss_to_box(box, twi: pd.DataFrame):
    """ Rough sanity check to ensure elegant twiss data matches internal calculations """
    df_t = box.twiss_df()
    df_t['name'] = df_t['name'].shift(1).str.lower()
    twi['ElementName'] = twi['ElementName'].str.lower()
    df_tc = df_t.drop_duplicates('name')
    twic = twi.drop_duplicates('ElementName')
    df_m = df_tc.merge(twic, right_on=['ElementName'], left_on=['name'])
    assert all(np.isclose(df_m.s_x, df_m.s_y))
    pairs = [('beta_x', 'betax'), ('mux', 'psix'), ('Dx', 'etax'), ('Dxp', 'etaxp'), ('beta_y', 'betay'),
             ('muy', 'psiy'), ('Dy', 'etay'), ('Dyp', 'etayp'), ]
    assert len(df_m) >= 1
    for p in pairs:
        v = (df_m.loc[:, p[0]] - df_m.loc[:, p[1]]).abs().max()
        if v > 1e-8:
            raise Exception(f'Twiss mismatch - failed on pair {p}, delta: {v}')


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


